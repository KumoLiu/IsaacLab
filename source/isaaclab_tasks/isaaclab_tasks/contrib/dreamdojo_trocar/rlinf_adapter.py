# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task-specific RLinf adapter for the robot-free DreamDojo environment.

One native subprocess shares a WM across all slots on its assigned GPU. This
integration supports fixed-horizon GRPO/evaluation with explicit resets.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rlinf.envs.isaaclab.venv import SubProcIsaacLabEnv


class TrocarRLinfEnv(gym.Env):
    """Adapt the DreamDojo G1 trocar task to RLinf, not arbitrary neural tasks.

    The 12-command/28-joint contract, recorded resets and milestone metrics are
    task-specific. Policy inference and optimization remain in RLinf.
    """

    @staticmethod
    def register_policy_converters() -> None:
        """Register task data mappings, without replacing RLinf's GR00T model."""
        from rlinf.models.embodiment.gr00t.simulation_io import ACTION_CONVERSION_N1D7, OBS_CONVERSION

        OBS_CONVERSION["g1_dex3_wm"] = TrocarRLinfEnv._gr00t_observation
        ACTION_CONVERSION_N1D7["g1_dex3_wm"] = TrocarRLinfEnv._gr00t_action

    @staticmethod
    def _gr00t_observation(obs: dict) -> dict:
        """Map head-view pixels and absolute joints [rad] to checkpoint modalities."""
        images = torch.as_tensor(obs["main_images"]).detach().cpu().numpy()
        states = torch.as_tensor(obs["states"]).detach().cpu().numpy()
        if states.ndim != 2 or states.shape[1] != 28:
            raise ValueError("Expected G1 state shaped (B,28)")
        result = {
            "video.head_view": images[:, None],
            "annotation.human.task_description": obs["task_descriptions"],
        }
        for index, name in enumerate(("left_arm", "right_arm", "left_hand", "right_hand")):
            result[f"state.{name}"] = states[:, None, index * 7 : (index + 1) * 7]
        return result

    @staticmethod
    def _gr00t_action(action: dict, chunk_size: int = 12) -> np.ndarray:
        """Concatenate already-decoded absolute targets [rad], without adding state."""
        prefix = "action." if "action.left_arm" in action else ""
        parts = [
            action[f"{prefix}{name}"][:, :chunk_size] for name in ("left_arm", "right_arm", "left_hand", "right_hand")
        ]
        if any(part.ndim != 3 or part.shape[1:] != (chunk_size, 7) for part in parts):
            raise ValueError("Each G1 action component must have shape (B,chunk_size,7)")
        return np.concatenate(parts, axis=-1)

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: object,
    ) -> None:
        del worker_info
        if cfg.auto_reset or cfg.enable_offload:
            raise ValueError("Native neural adapter requires auto_reset=false, enable_offload=false")
        if not cfg.get("is_eval", False) and cfg.ignore_terminations:
            raise ValueError("GRPO training requires ignore_terminations=false so RLinf builds its loss mask")
        if num_envs < 1 or cfg.group_size < 1 or num_envs % cfg.group_size:
            raise ValueError("num_envs must be positive and divisible by group_size")
        if (
            cfg.max_episode_steps != cfg.inference.max_chunks * 12
            or cfg.max_steps_per_rollout_epoch != cfg.max_episode_steps
        ):
            raise ValueError(
                "Use a complete rollout: max_episode_steps = max_steps_per_rollout_epoch = max_chunks * 12"
            )
        dataset = Path(cfg.inference.dataset)
        episode_count = json.loads((dataset / "meta/info.json").read_text())["total_episodes"]
        self.episode_ids = list(cfg.reset_episode_ids)
        if not self.episode_ids or any(type(i) is not int or not 0 <= i < episode_count for i in self.episode_ids):
            raise ValueError("reset_episode_ids must contain valid dataset episode indices")
        self.cfg, self.num_envs = cfg, num_envs
        self.seed = int(cfg.seed) + seed_offset  # Unique video filename per RLinf worker.
        self._base_seed = int(cfg.seed)
        self._offset = seed_offset * num_envs
        self._total_groups = total_num_processes * num_envs // cfg.group_size
        self._round = 0
        self.is_start, self.auto_reset = True, False
        self.device = torch.device("cpu")
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (12, 28), np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "main_images": gym.spaces.Box(0, 255, (480, 640, 3), np.uint8),
                "states": gym.spaces.Box(-np.inf, np.inf, (28,), np.float32),
            }
        )
        self._worker = None
        self._needs_reset = True
        self._frames = None
        log_dir = Path(cfg.native_log_dir).resolve() / f"rank_{seed_offset}_{uuid.uuid4().hex[:8]}"
        inference = OmegaConf.to_container(cfg.inference, resolve=True)
        for key in (
            "dreamdojo_root",
            "dataset",
            "wm_checkpoint",
            "reward_checkpoint",
            "lam_checkpoint",
        ):
            inference[key] = str(Path(inference[key]).absolute())
        # Respect RLinf/Ray's assigned physical device, never hard-code GPU 0.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        if len(visible) != 1 or not visible[0]:
            raise ValueError("Assign exactly one GPU to each RLinf environment worker")

        def make_env():
            # The child owns local inference, not a distributed policy rank.
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                os.environ.pop(key, None)
            from isaaclab_contrib.neural.neural_env import NeuralDirectEnv

            from isaaclab_tasks.contrib.dreamdojo_trocar.env_cfg import (
                DreamDojoInferenceCfg,
                make_dreamdojo_env_cfg,
            )

            return NeuralDirectEnv(make_dreamdojo_env_cfg(DreamDojoInferenceCfg(**inference), num_envs=num_envs))

        self._worker = SubProcIsaacLabEnv(
            make_env, log_path=str(log_dir / "native.log"), timeout_s=float(cfg.get("worker_timeout_s", 600.0))
        )

    def _wrap_obs(self, observation: dict, info: dict) -> dict:
        self._frames = torch.from_numpy(info["frames"])
        return observation | {
            "wrist_images": None,
            "task_descriptions": info["task_description"],
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset all slots to group-shared cases and independent rollout seeds."""
        if options:
            raise ValueError("Select cases through reset_episode_ids, not reset options")
        if self._worker is None:
            raise RuntimeError("Native environment is closed")
        if seed is not None:
            self._base_seed, self._round = int(seed), 0
        episodes, seeds = [], []
        for slot in range(self.num_envs):
            group = (self._offset + slot) // self.cfg.group_size
            episodes.append(self.episode_ids[(group + self._round) % len(self.episode_ids)])
            # Same case and WM noise within each GRPO group; RLinf owns policy exploration.
            seeds.append((self._base_seed + group + self._round * self._total_groups) % 2**32)
        try:
            native_obs, extras = self._worker.reset(
                seed=self._base_seed, options={"episode_indices": episodes, "seeds": seeds}
            )
            obs = self._wrap_obs(native_obs["policy"], extras["reset_info"])
        except BaseException:
            self.close()
            raise
        self._needs_reset = False
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int64)
        self._returns = torch.zeros(self.num_envs)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, {}

    def chunk_step(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """Execute absolute targets [rad], `(B,12,28)`, once per native environment."""
        if self._needs_reset:
            raise RuntimeError("Reset before starting or continuing a completed rollout")
        actions = torch.as_tensor(actions).detach().cpu().float().numpy()
        if actions.shape != (self.num_envs, 12, 28) or not np.isfinite(actions).all():
            raise ValueError("Expected finite absolute joint targets shaped (num_envs,12,28)")
        try:
            obs, reward, terminated, truncated, extras = self._worker.step(torch.from_numpy(actions))
            info = extras["neural_transition"]
            # Native final_obs contains the complete pre-reset batch, including live rows.
            terminal_obs = extras.get("final_obs", obs)
            obs = self._wrap_obs(terminal_obs["policy"], info)
            rewards = torch.from_numpy(info["command_rewards"]).float()
            if not torch.allclose(rewards.sum(dim=1), reward):
                raise ValueError("Native reward must equal the sum of command-level rewards")
            rewards = rewards * self.cfg.reward_coef
            stages = torch.as_tensor(info["milestone_stage"])
            truncations = torch.zeros(self.num_envs, 12, dtype=torch.bool)
            truncations[:, -1] = truncated
            if terminated.any():
                raise RuntimeError("Fixed-horizon native task unexpectedly terminated early")
        except BaseException:
            self.close()
            raise
        self._elapsed_steps += 12
        self._returns += rewards.sum(dim=1)
        self._success |= torch.as_tensor(info["is_success"])
        self._needs_reset = bool(truncations.any())
        episode = {
            "success_once": self._success.clone(),
            "return": self._returns.clone(),
            "episode_len": self._elapsed_steps.clone(),
            "episode_seconds": self._elapsed_steps / 30,
            "reward": self._returns / self._elapsed_steps,
        }
        for index, name in enumerate(("picked", "handed", "placed")):
            episode[name] = stages > index
        return (
            [obs],
            rewards,
            torch.zeros_like(truncations),
            truncations,
            [{"episode": episode}],
        )

    def update_reset_state_ids(self) -> None:
        """Advance deterministic case selection once per RLinf rollout."""
        if not self.cfg.use_fixed_reset_state_ids:
            self._round += 1

    def capture_image(self) -> torch.Tensor | None:
        """Expose all six generated frames per chunk to RLinf's native recorder."""
        return self._frames

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    def close(self) -> None:
        """Close the shared native context/WM process."""
        self._needs_reset = True
        if self._worker is not None:
            self._worker.close()
            self._worker = None
