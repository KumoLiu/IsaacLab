# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-process Cosmos3 dynamics using native preprocessing, sampling and sessions.

No policy, robot kinematics, action normalization or task reward lives here.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from cosmos_framework.inference.bimanual import BimanualActionContract
    from cosmos_framework.inference.bimanual_session import BimanualSession
    from cosmos_framework.inference.inference import OmniInference

    from .env_cfg import Cosmos3InferenceCfg


class Cosmos3Simulator:
    """Share model weights across independent windows, or drive one native causal session."""

    def __init__(self, cfg: Cosmos3InferenceCfg, num_envs: int):
        self.cfg = cfg
        self.num_envs = num_envs
        self.pipe: OmniInference | None = None
        self.session: BimanualSession | None = None
        self.contract: BimanualActionContract | None = None
        self.condition: list[torch.Tensor | None] = [None] * num_envs
        self.prompts = [""] * num_envs
        self.seeds = np.zeros(num_envs, np.int64)
        self.elapsed = np.zeros(num_envs, np.int64)

    def load(self) -> None:
        """Load a local HF export or a DCP checkpoint with its saved YAML config."""
        self.cfg.validate()
        # The neural backend samples frozen dynamics; Framework's default enables
        # its separate world-model training dependency stack at import time.
        os.environ.setdefault("COSMOS_TRAINING", "0")
        try:
            from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
            from cosmos_framework.inference.args import OmniSetupOverrides
            from cosmos_framework.inference.bimanual import BimanualActionContract
        except ModuleNotFoundError as error:
            if error.name == "cosmos_framework":
                raise ModuleNotFoundError(
                    "Cosmos3 requires Cosmos Framework in the active Isaac Lab environment. "
                    "Run 'uv sync --extra cosmos3' as documented in "
                    "source/isaaclab_contrib/docs/neural-simulator.md."
                ) from error
            raise
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError("Cosmos3 neural environments require one inference process per model")
        kwargs = {"config_file": self.cfg.config_file} if self.cfg.config_file is not None else {}
        setup = OmniSetupOverrides(
            checkpoint_path=self.cfg.checkpoint_path,
            output_dir=self.cfg.output_dir,
            use_ema_weights=self.cfg.use_ema_weights,
            guardrails=False,
            use_torch_compile=False,
            use_cuda_graphs=False,
            diffusion_cache=False,
            dp_replicate_size=1,
            dp_shard_size=1,
            cp_size=1,
            cfgp_size=1,
            **kwargs,
        ).build_setup()
        self.pipe = setup.get_inference_cls().create(setup)
        model = self.pipe.model
        if model.config.action_schema != "unified_v1" or model.config.max_action_dim != 64:
            raise ValueError("Select a unified_v1 Cosmos3 Bimanual checkpoint (59 controls padded to 64)")
        if self.cfg.mode == "window" and model.config.video_temporal_causal:
            raise ValueError("A causal checkpoint requires mode='session' and its saved distilled sampler")
        self.contract = getattr(model, "_bimanual_action_contract", None)
        if self.contract is None:
            self.contract = BimanualActionContract(59, 64, {self.cfg.domain_name: get_domain_id(self.cfg.domain_name)})
        if self.cfg.domain_name not in self.contract.domain_ids:
            raise ValueError(f"Checkpoint does not support domain {self.cfg.domain_name!r}")

    def _pixels(self, rgb: torch.Tensor, count: int) -> np.ndarray:
        expected = (1, 3, count, *self.cfg.image_size)
        if tuple(rgb.shape) != expected or not torch.isfinite(rgb).all():
            raise ValueError(f"Cosmos3 decoded RGB must be finite with shape {expected}; got {tuple(rgb.shape)}")
        # Match the native public API before using pixels as the next condition.
        return rgb.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8).permute(0, 2, 3, 4, 1).numpy()

    def _batch(self, condition: torch.Tensor, action: np.ndarray, prompt: str) -> dict:
        from cosmos_framework.inference.action import build_action_batch
        from cosmos_framework.inference.args import ModelMode
        from cosmos_framework.inference.inference import _finalize_data_batch

        payload = {"action_space": "normalized_unified_v1", "action": action.tolist()}
        values, domains = self.contract.validate(
            payload, action_count=self.cfg.chunk_size, domain_name=self.cfg.domain_name
        )
        control = torch.zeros(self.cfg.chunk_size, 64, dtype=torch.float32)
        control[:, :59] = torch.tensor(values, dtype=torch.float32)
        batch = build_action_batch(
            video=condition,
            action=control,
            raw_action_dim=59,
            prompt=prompt,
            view_point="ego_view",
            domain_name=self.cfg.domain_name,
            model_mode=ModelMode.FORWARD_DYNAMICS,
            action_chunk_size=self.cfg.chunk_size,
            fps=self.cfg.fps,
            resolution=self.cfg.resolution,
            input_video_key=self.pipe.model.input_video_key,
            device=self.pipe.model.tensor_kwargs["device"],
            action_domain_ids=torch.tensor(domains, dtype=torch.long),
            format_prompt_as_json=False,
        )
        return _finalize_data_batch(batch, 1, self.pipe.model)

    @torch.inference_mode()
    def reset(self, env_ids: np.ndarray, images: list[Path], prompts: list[str], seeds: np.ndarray) -> np.ndarray:
        """Reset selected rows from real images; untouched rows retain their generated conditions."""
        if self.pipe is None:
            raise RuntimeError("Load Cosmos3 before resetting")
        if self.cfg.mode == "session":
            if self.session is not None:
                self.session.close()
            self.session = self.pipe.open_bimanual_session(
                vision_path=images[0],
                prompt=prompts[0],
                domain_name=self.cfg.domain_name,
                fps=self.cfg.fps,
                seed=int(seeds[0]),
                resolution=self.cfg.resolution,
            )
            if self.session.action_rows != self.cfg.chunk_size:
                raise ValueError(f"Checkpoint requires chunk_size={self.session.action_rows}")
            frames = self._pixels(self.session.observation, 1)
        else:
            from cosmos_framework.inference.vision import read_media_frames

            prepared = []
            for index in env_ids:
                condition, _ = read_media_frames(images[index], max_frames=1)
                prepared.append(self._pixels(condition[None].float() / 255, 1))
            frames = np.concatenate(prepared)
        for index, frame in zip(env_ids, frames, strict=True):
            self.condition[index] = torch.from_numpy(frame[-1].copy()).permute(2, 0, 1)[:, None]
            self.prompts[index] = prompts[index]
        self.seeds[env_ids] = seeds[env_ids]
        self.elapsed[env_ids] = 0
        return frames

    @torch.inference_mode()
    def step(self, action: np.ndarray) -> np.ndarray:
        """Predict only from generated history and the caller's prepared normalized controls."""
        if self.pipe is None or any(c is None for c in self.condition):
            raise RuntimeError("Reset all environments before applying actions")
        expected = (self.num_envs, self.cfg.chunk_size, 59)
        if action.shape != expected or not np.isfinite(action).all():
            raise ValueError(f"Expected finite normalized controls shaped {expected}")
        if self.cfg.mode == "session":
            rgb = self.session.step({"action_space": "normalized_unified_v1", "action": action[0].tolist()})
            frames = self._pixels(rgb, self.cfg.chunk_size)
        else:
            prepared = [self._batch(self.condition[i], action[i], self.prompts[i]) for i in range(self.num_envs)]
            chunks = []
            for index, batch in enumerate(prepared):
                result = self.pipe.model.generate_samples_from_batch(
                    batch,
                    guidance=self.cfg.guidance,
                    shift=10.0 if self.cfg.shift is None else self.cfg.shift,
                    seed=[int((self.seeds[index] + self.elapsed[index]) % 2**32)],
                    num_steps=self.cfg.num_inference_steps or 30,
                    n_sample=1,
                )
                rgb = (self.pipe.model.decode(result["vision"][0]) + 1) / 2
                chunks.append(self._pixels(rgb, self.cfg.chunk_size + 1)[:, 1:])
            frames = np.concatenate(chunks)
        for index, frame in enumerate(frames):
            self.condition[index] = torch.from_numpy(frame[-1].copy()).permute(2, 0, 1)[:, None]
        self.elapsed += self.cfg.chunk_size
        return frames

    def info(self, env_ids: np.ndarray | None = None) -> dict:
        """Report generated time [s] and model history, without inventing physical joint states."""
        ids = np.arange(self.num_envs) if env_ids is None else env_ids
        info = {
            "model_only": True,
            "mode": self.cfg.mode,
            "elapsed_frames": self.elapsed[ids].copy(),
            "elapsed_seconds": self.elapsed[ids] / self.cfg.fps,
            "task_description": [self.prompts[i] for i in ids],
        }
        if self.session is not None:
            info["session"] = self.session.inspect()
        return info

    def close(self) -> None:
        """Release this simulator's decoder/cache and model references, without touching other backends."""
        try:
            if self.session is not None:
                self.session.close()
        finally:
            self.session = None
            self.pipe = None
            self.contract = None
            self.condition = [None] * self.num_envs
