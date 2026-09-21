# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful G1 trocar simulation using DreamDojo and a reward classifier.

Recorded resets, G1 conditioning and rewards are owned by this task package.
Task lifecycle belongs to Isaac Lab; this module does not create processes.
"""

from __future__ import annotations

import gc
import os
from contextlib import chdir
from pathlib import Path

import numpy as np
import torch


class DreamDojoSimulator:
    """Manage task state, action-conditioned transitions and rewards, without a Gym environment."""

    def __init__(self, settings: dict):
        from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

        from .actions import G1DreamDojoActionBridge
        from .dataset import LeRobotV21InitDataset
        from .reward import BatchedMilestoneReward

        self.settings = settings
        self.num_envs = settings.get("num_envs", 1)
        self.dataset = LeRobotV21InitDataset(settings["dataset"])
        self.bridge = G1DreamDojoActionBridge(Path(settings["dreamdojo_root"]) / "shared_meta/G1_stats.json", "cuda")
        # DreamDojo resolves some support assets relative to its checkout.
        # Restore process state afterwards; no subprocess is needed for inference.
        previous_lam = os.environ.get("DREAMDOJO_LAM_CHECKPOINT")
        try:
            os.environ["DREAMDOJO_LAM_CHECKPOINT"] = settings["lam_checkpoint"]
            with chdir(settings["dreamdojo_root"]):
                self.pipe = Video2WorldInference(
                    experiment_name=settings["experiment"],
                    ckpt_path=settings["wm_checkpoint"],
                    s3_credential_path="",
                    context_parallel_size=1,
                    config_file="isaaclab_tasks/contrib/dreamdojo_trocar/env_cfg.py",
                    offload_diffusion_model=False,
                    offload_text_encoder=True,
                    offload_tokenizer=False,
                )
        finally:
            if previous_lam is None:
                os.environ.pop("DREAMDOJO_LAM_CHECKPOINT", None)
            else:
                os.environ["DREAMDOJO_LAM_CHECKPOINT"] = previous_lam
        self.reward_model = BatchedMilestoneReward(
            settings["reward_checkpoint"], self.num_envs, "cuda", duplicate_for_30fps=True
        )
        self.ready = np.zeros(self.num_envs, dtype=bool)
        self.seed = np.zeros(self.num_envs, dtype=np.int64)
        self.episode = np.zeros(self.num_envs, dtype=np.int64)
        self.start_frame = np.zeros(self.num_envs, dtype=np.int64)
        self.elapsed_commands = np.zeros(self.num_envs, dtype=np.int64)
        self.task = [""] * self.num_envs
        print(f"In-process DreamDojo inference: pid={os.getpid()}", flush=True)

    @torch.inference_mode()
    def reset(self, env_ids: np.ndarray, episodes: np.ndarray, seeds: np.ndarray) -> dict[str, np.ndarray]:
        """Initialize selected rows from recorded frame 2, preserving all other histories."""
        if np.any(episodes < 0) or np.any(episodes >= len(self.dataset)):
            raise ValueError(f"Invalid episode indices: {episodes}")
        items = [self.dataset[int(episode)] for episode in episodes]
        indices = torch.as_tensor(env_ids, device="cuda")
        for name, key in (
            ("condition", "image"),
            ("state", "state"),
            ("previous_action", "previous_action"),
            ("previous_state", "previous_state"),
        ):
            values = torch.stack([item[key] for item in items]).to("cuda")
            if name == "condition":
                values = values.sub(0.5).div(0.5)
            if not hasattr(self, name):
                setattr(self, name, values.new_zeros((self.num_envs, *values.shape[1:])))
            getattr(self, name)[indices] = values
        self.seed[env_ids], self.episode[env_ids] = seeds, episodes
        self.start_frame[env_ids] = [int(item["start_frame"]) for item in items]
        for index, item in zip(env_ids, items, strict=True):
            self.task[index] = str(item["task"])
        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device="cuda")
        reset_mask[indices] = True
        self.reward_model.reset(reset_mask)
        self.elapsed_commands[env_ids] = 0
        self.ready[env_ids] = True
        observation = self._observation(env_ids)
        return {**observation, "frames": observation["main_images"][:, None]}

    @torch.inference_mode()
    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """Advance one chunk using generated-image feedback and unchanged commands."""
        if not self.ready.all():
            raise RuntimeError("reset() required before step()")
        if action.shape != (self.num_envs, 12, 28) or not np.isfinite(action).all():
            raise ValueError("Invalid action payload")
        actions = torch.as_tensor(action, device="cuda", dtype=torch.float32)
        encoded = self.bridge.encode_policy_chunk(self.previous_action, actions, self.previous_state)
        condition = ((self.condition + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        video = torch.cat(
            [condition[:, :, None], torch.zeros_like(condition)[:, :, None].repeat(1, 1, 12, 1, 1)], dim=2
        )
        sample_seeds = ((self.seed + self.elapsed_commands) % 2**32).tolist()
        generated = self.pipe.generate_vid2world(
            prompt="",
            input_path=video,
            action=encoded,
            guidance=0,
            num_video_frames=13,
            num_latent_conditional_frames=1,
            resolution="480,640",
            seed=sample_seeds[0],
            sample_seeds=sample_seeds,
            lam_video=None,
            num_steps=self.settings["num_inference_steps"],
        )
        frames = generated[:, :, 1:7]
        frame_rewards, probabilities = self.reward_model.score_chunk(frames.permute(0, 2, 1, 3, 4))
        rewards = frame_rewards.new_zeros(self.num_envs, 12)
        rewards[:, 1::2] = frame_rewards
        self.condition = frames[:, :, -1]
        self.previous_action = actions[:, -2].detach()
        self.previous_state = actions[:, -3].detach().clone()
        self.state = actions[:, -1].clone()
        self.elapsed_commands += 12
        # Report the model horizon; Isaac Lab performs the actual episode reset.
        truncated = self.elapsed_commands >= self.settings["max_chunks"] * 12
        self.ready = ~truncated
        return {
            **self._observation(),
            "frames": ((frames + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 4, 1).cpu().numpy(),
            "probabilities": probabilities.cpu().numpy(),
            "command_rewards": rewards.cpu().numpy(),
            "reward": rewards.sum(dim=1).cpu().numpy(),
            "terminated": np.zeros(self.num_envs, dtype=bool),
            "truncated": truncated,
        }

    def _observation(self, env_ids: np.ndarray | None = None) -> dict[str, np.ndarray]:
        indices = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.state.device)
        return {
            "main_images": ((self.condition[indices] + 1.0) / 2.0 * 255.0)
            .clamp(0, 255)
            .permute(0, 2, 3, 1)
            .to(torch.uint8)
            .cpu()
            .numpy(),
            "states": self.state[indices].cpu().numpy(),
        }

    def info(self, env_ids: np.ndarray | None = None) -> dict:
        """Return time alignment and model predictions, not ground-truth success."""
        indices = np.arange(self.num_envs) if env_ids is None else env_ids
        stage = self.reward_model.stage.cpu().numpy()[indices].copy()
        return {
            "episode_index": self.episode[indices].copy(),
            "start_frame": self.start_frame[indices].copy(),
            "frame_index_30hz": (self.start_frame + self.elapsed_commands)[indices],
            "elapsed_commands": self.elapsed_commands[indices].copy(),
            "elapsed_seconds": self.elapsed_commands[indices] / 30,
            "task_description": [self.task[i] for i in indices],
            "milestone_stage": stage,
            "is_success": stage == 3,
            "model_only": True,
        }

    def close(self) -> None:
        """Release model weights and episode tensors without exiting the process."""
        self.__dict__.clear()
        self.ready = np.array([False])
        gc.collect()
        torch.cuda.empty_cache()
