# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an IsaacLab environment for RLinf library.

This wrapper bridges IsaacLab's environment interface with RLinf's expected interface,
handling observation format conversion and providing the necessary APIs for RLinf's
distributed training architecture.

The following example shows how to wrap an environment for RLinf:

.. code-block:: python

    from rlinf_wrapper import RLinfVecEnvWrapper

    # Create IsaacLab environment
    env = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0", cfg=env_cfg)

    # Wrap for RLinf
    env = RLinfVecEnvWrapper(
        env,
        task_description="Pick up the red cube and place it on top of the blue cube.",
        obs_keys_mapping={
            "main_images": "table_cam",
            "wrist_images": "wrist_cam",
            "states": ["eef_pos", "eef_quat", "gripper_pos"],
        }
    )
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


def quat_wxyz_to_axisangle(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to axis-angle representation.

    Args:
        quat: Quaternion tensor of shape (..., 4) in wxyz format.

    Returns:
        Axis-angle tensor of shape (..., 3).
    """
    # Convert wxyz to xyzw for computation
    quat_xyzw = quat[..., [1, 2, 3, 0]]

    # Clamp w component
    w = torch.clamp(quat_xyzw[..., 3:4], -1.0, 1.0)
    xyz = quat_xyzw[..., :3]

    # Compute angle
    angle = 2.0 * torch.acos(w)

    # Compute axis
    sin_half_angle = torch.sqrt(1.0 - w * w)
    # Avoid division by zero
    mask = sin_half_angle.abs() < 1e-8
    sin_half_angle = torch.where(mask, torch.ones_like(sin_half_angle), sin_half_angle)

    axis = xyz / sin_half_angle
    axis = torch.where(mask.expand_as(axis), torch.zeros_like(axis), axis)

    # Return axis-angle (axis * angle)
    return axis * angle


class RLinfVecEnvWrapper:
    """Wraps around IsaacLab environment for the RLinf library.

    This wrapper converts IsaacLab's observation and action formats to match
    RLinf's expected interface for embodied AI training.

    RLinf expects observations in the format:
        {
            "main_images": (B, H, W, C) or (B, C, H, W) tensor,
            "wrist_images": (B, H, W, C) or (B, C, H, W) tensor (optional),
            "states": (B, state_dim) tensor,
            "task_descriptions": list of strings,
        }

    Attributes:
        env: The wrapped IsaacLab environment.
        num_envs: Number of parallel environments.
        device: Device where tensors are stored.
        task_description: Task description string for all environments.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv | gym.Env,
        task_description: str = "",
        obs_keys_mapping: dict[str, str | list[str]] | None = None,
        state_keys: list[str] | None = None,
        image_format: str = "channels_first",  # "channels_first" (C, H, W) or "channels_last" (H, W, C)
        convert_quat_to_axisangle: bool = True,
        clip_actions: float | None = None,
    ):
        """Initialize the RLinf wrapper.

        Args:
            env: The IsaacLab environment to wrap.
            task_description: Task description string used for all environments.
            obs_keys_mapping: Mapping from RLinf obs keys to IsaacLab obs keys.
                Example: {"main_images": "table_cam", "wrist_images": "wrist_cam"}
            state_keys: List of state observation keys to concatenate.
                Example: ["eef_pos", "eef_quat", "gripper_pos"]
            image_format: Format of image observations ("channels_first" or "channels_last").
            convert_quat_to_axisangle: Whether to convert quaternion to axis-angle.
            clip_actions: Optional action clipping value.
        """
        # Validate environment type
        if not isinstance(env.unwrapped, (ManagerBasedRLEnv, DirectRLEnv)):
            raise ValueError(
                f"Environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. "
                f"Got: {type(env.unwrapped)}"
            )

        self.env = env
        self.task_description = task_description
        self.obs_keys_mapping = obs_keys_mapping or {}
        self.state_keys = state_keys or []
        self.image_format = image_format
        self.convert_quat_to_axisangle = convert_quat_to_axisangle
        self.clip_actions = clip_actions

        # Store environment properties
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # Get action dimension
        if hasattr(self.unwrapped, "action_manager"):
            self.action_dim = self.unwrapped.action_manager.total_action_dim
        else:
            self.action_dim = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # Initialize tracking variables
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._success_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment."""
        return self.env.unwrapped

    @property
    def observation_space(self) -> gym.Space:
        """Returns the observation space."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the action space."""
        return self.env.action_space

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """Returns elapsed steps for each environment."""
        return self._elapsed_steps

    def reset(
        self,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional seed for reset.
            env_ids: Optional tensor of environment indices to reset.

        Returns:
            Tuple of (observations, info).
        """
        if env_ids is not None:
            # Partial reset - IsaacLab handles this internally
            # We need to trigger reset for specific environments
            # This is typically handled by the step function in IsaacLab
            pass

        obs_dict, info = self.env.reset(seed=seed)

        # Reset tracking for specified environments
        if env_ids is not None:
            self._elapsed_steps[env_ids] = 0
            self._returns[env_ids] = 0.0
            self._success_once[env_ids] = False
        else:
            self._elapsed_steps.zero_()
            self._returns.zero_()
            self._success_once.zero_()

        # Convert observations to RLinf format
        rlinf_obs = self._convert_obs(obs_dict)

        return rlinf_obs, info

    def step(
        self,
        actions: torch.Tensor,
        auto_reset: bool = False,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment.

        Args:
            actions: Actions tensor of shape (num_envs, action_dim).
            auto_reset: Whether to auto-reset terminated environments.

        Returns:
            Tuple of (observations, rewards, terminations, truncations, info).
        """
        # Clip actions if specified
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # Step the environment
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)

        # Update tracking
        self._elapsed_steps += 1
        self._returns += rewards
        self._success_once = self._success_once | terminated

        # Convert observations to RLinf format
        rlinf_obs = self._convert_obs(obs_dict)

        # Add episode info
        info["episode"] = {
            "success_once": self._success_once.clone(),
            "return": self._returns.clone(),
            "episode_len": self._elapsed_steps.clone(),
        }

        # Handle auto reset
        dones = terminated | truncated
        if auto_reset and dones.any():
            # Store final observations
            info["final_observation"] = copy.deepcopy(rlinf_obs)
            info["final_info"] = copy.deepcopy(info)
            info["_final_observation"] = dones
            info["_final_info"] = dones

            # Reset terminated environments
            done_indices = torch.where(dones)[0]
            rlinf_obs, _ = self.reset(env_ids=done_indices)

        return rlinf_obs, rewards, terminated, truncated, info

    def chunk_step(
        self,
        chunk_actions: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Execute a chunk of actions.

        Args:
            chunk_actions: Actions tensor of shape (num_envs, chunk_size, action_dim).

        Returns:
            Tuple of (final_obs, chunk_rewards, chunk_terminations, chunk_truncations, info).
        """
        chunk_size = chunk_actions.shape[1]

        chunk_rewards = []
        chunk_terminations = []
        chunk_truncations = []

        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            obs, rewards, terminated, truncated, info = self.step(actions, auto_reset=False)
            chunk_rewards.append(rewards)
            chunk_terminations.append(terminated)
            chunk_truncations.append(truncated)

        # Stack results
        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        chunk_terminations = torch.stack(chunk_terminations, dim=1)
        chunk_truncations = torch.stack(chunk_truncations, dim=1)

        return obs, chunk_rewards, chunk_terminations, chunk_truncations, info

    def _convert_obs(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert IsaacLab observations to RLinf format.

        Args:
            obs_dict: Observations from IsaacLab in format {"policy": {...}}.

        Returns:
            Observations in RLinf format.
        """
        rlinf_obs = {}

        # Get the policy observations
        if "policy" in obs_dict:
            policy_obs = obs_dict["policy"]
        else:
            policy_obs = obs_dict

        # Convert images
        for rlinf_key, lab_key in self.obs_keys_mapping.items():
            if lab_key in policy_obs:
                image = policy_obs[lab_key]
                # Convert image format if needed
                if self.image_format == "channels_last" and image.dim() == 4:
                    # From (B, C, H, W) to (B, H, W, C)
                    image = image.permute(0, 2, 3, 1)
                rlinf_obs[rlinf_key] = image

        # Convert states
        if self.state_keys:
            state_parts = []
            for key in self.state_keys:
                if key in policy_obs:
                    state = policy_obs[key]
                    # Convert quaternion to axis-angle if needed
                    if self.convert_quat_to_axisangle and "quat" in key.lower():
                        state = quat_wxyz_to_axisangle(state)
                    state_parts.append(state)

            if state_parts:
                rlinf_obs["states"] = torch.cat(state_parts, dim=-1)

        # Add task descriptions
        rlinf_obs["task_descriptions"] = [self.task_description] * self.num_envs

        return rlinf_obs

    def close(self):
        """Close the environment."""
        return self.env.close()

    def seed(self, seed: int) -> int:
        """Set the random seed."""
        return self.unwrapped.seed(seed)


class RLinfIsaacLabEnv:
    """A wrapper class that creates an IsaacLab environment compatible with RLinf.

    This class follows the RLinf environment interface (similar to IsaaclabBaseEnv)
    while using IsaacLab's gym.make() for environment creation.

    This allows IsaacLab registered tasks to be used directly with RLinf's training
    infrastructure without needing to manually register each task in REGISTER_ISAACLAB_ENVS.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info,
    ):
        """Initialize the environment.

        Args:
            cfg: RLinf environment configuration (OmegaConf).
            num_envs: Number of parallel environments.
            seed_offset: Seed offset for this worker.
            total_num_processes: Total number of worker processes.
            worker_info: Worker information from RLinf scheduler.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info

        # Environment configuration
        self.task_id = cfg.init_params.id
        self.task_description = cfg.init_params.get("task_description", "")
        self.max_episode_steps = cfg.max_episode_steps
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations

        # Video configuration
        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.images = []

        # Observation mapping configuration
        self.obs_keys_mapping = cfg.init_params.get("obs_keys_mapping", {})
        self.state_keys = cfg.init_params.get("state_keys", [])

        # Initialize the environment
        self._init_env()

        # Initialize metrics
        self._init_metrics()

    def _init_env(self):
        """Initialize the IsaacLab environment."""
        from rlinf.envs.isaaclab.venv import SubProcIsaacLabEnv

        def make_env():
            from isaaclab.app import AppLauncher

            # Launch simulator
            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            # Import after AppLauncher
            import gymnasium as gym
            from isaaclab_tasks.utils import load_cfg_from_registry

            # Load and configure environment
            env_cfg = load_cfg_from_registry(self.task_id, "env_cfg_entry_point")
            env_cfg.scene.num_envs = self.num_envs

            # Apply any custom configuration from init_params
            if hasattr(self.cfg.init_params, "scene"):
                for key, value in self.cfg.init_params.scene.items():
                    if hasattr(env_cfg.scene, key):
                        setattr(env_cfg.scene, key, value)

            # Create environment
            env = gym.make(self.task_id, cfg=env_cfg, render_mode="rgb_array").unwrapped

            return env, sim_app

        # Create subprocess environment
        self.env = SubProcIsaacLabEnv(make_env)
        self.env.reset(seed=self.seed)
        self.device = self.env.device()

    def _init_metrics(self):
        """Initialize tracking metrics."""
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._success_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_step_reward = torch.zeros(self.num_envs, device=self.device)

    def _reset_metrics(self, env_idx=None):
        """Reset metrics for specified environments."""
        if env_idx is not None:
            self._elapsed_steps[env_idx] = 0
            self._returns[env_idx] = 0.0
            self._success_once[env_idx] = False
            self.prev_step_reward[env_idx] = 0.0
        else:
            self._elapsed_steps.zero_()
            self._returns.zero_()
            self._success_once.zero_()
            self.prev_step_reward.zero_()

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    def reset(self, seed: int | None = None, env_ids: torch.Tensor | None = None):
        """Reset the environment."""
        if env_ids is None:
            obs, _ = self.env.reset(seed=seed)
        else:
            obs, _ = self.env.reset(seed=seed, env_ids=env_ids)

        self._reset_metrics(env_ids)
        obs = self._wrap_obs(obs)

        return obs, {}

    def step(self, actions: torch.Tensor, auto_reset: bool = True):
        """Step the environment."""
        obs, step_reward, terminations, truncations, infos = self.env.step(actions)

        if self.video_cfg.save_video:
            self.images.append(self._add_image(obs))

        obs = self._wrap_obs(obs)
        self._elapsed_steps += 1

        # Apply truncation based on max episode steps
        truncations = (self._elapsed_steps >= self.max_episode_steps) | truncations
        dones = terminations | truncations

        # Record metrics
        self._returns += step_reward
        self._success_once = self._success_once | terminations

        infos = self._record_metrics(step_reward, terminations, infos)

        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations.clone()
            terminations = torch.zeros_like(terminations)

        # Handle auto reset
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions: torch.Tensor):
        """Execute a chunk of actions."""
        chunk_size = chunk_actions.shape[1]

        chunk_rewards = []
        chunk_terminations = []
        chunk_truncations = []

        for i in range(chunk_size):
            obs, reward, term, trunc, infos = self.step(
                chunk_actions[:, i], auto_reset=False
            )
            chunk_rewards.append(reward)
            chunk_terminations.append(term)
            chunk_truncations.append(trunc)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        chunk_terminations = torch.stack(chunk_terminations, dim=1)
        chunk_truncations = torch.stack(chunk_truncations, dim=1)

        # Handle auto reset at end of chunk
        past_dones = (chunk_terminations | chunk_truncations).any(dim=1)
        if past_dones.any() and self.auto_reset:
            obs, infos = self._handle_auto_reset(past_dones, obs, infos)

        return obs, chunk_rewards, chunk_terminations, chunk_truncations, infos

    def _wrap_obs(self, obs: dict) -> dict:
        """Convert IsaacLab observations to RLinf format.

        Override this method for task-specific observation conversion.
        """
        rlinf_obs = {
            "task_descriptions": [self.task_description] * self.num_envs,
        }

        policy_obs = obs.get("policy", obs)

        # Map images
        for rlinf_key, lab_key in self.obs_keys_mapping.items():
            if lab_key in policy_obs:
                rlinf_obs[rlinf_key] = policy_obs[lab_key]

        # Build state vector
        if self.state_keys:
            state_parts = []
            for key in self.state_keys:
                if key in policy_obs:
                    state = policy_obs[key]
                    # Convert quaternion wxyz to axis-angle
                    if "quat" in key.lower():
                        state = quat_wxyz_to_axisangle(state)
                    state_parts.append(state)
            if state_parts:
                rlinf_obs["states"] = torch.cat(state_parts, dim=-1)

        return rlinf_obs

    def _add_image(self, obs: dict):
        """Extract image for video recording."""
        policy_obs = obs.get("policy", obs)
        main_image_key = self.obs_keys_mapping.get("main_images", "rgb")
        if main_image_key in policy_obs:
            return policy_obs[main_image_key][0].cpu().numpy()
        return None

    def _record_metrics(self, step_reward, terminations, infos):
        """Record episode metrics."""
        infos["episode"] = {
            "success_once": self._success_once.clone(),
            "return": self._returns.clone(),
            "episode_len": self._elapsed_steps.clone(),
            "reward": self._returns / (self._elapsed_steps + 1e-8),
        }
        return infos

    def _handle_auto_reset(self, dones, final_obs, infos):
        """Handle automatic reset for done environments."""
        final_obs = copy.deepcopy(final_obs)
        final_info = copy.deepcopy(infos)

        done_indices = torch.where(dones)[0]
        obs, _ = self.reset(env_ids=done_indices)

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_observation"] = dones
        infos["_final_info"] = dones

        return obs, infos

    def flush_video(self, video_sub_dir: str | None = None):
        """Save recorded video."""
        import os

        import imageio

        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir:
            output_dir = os.path.join(output_dir, video_sub_dir)
        os.makedirs(output_dir, exist_ok=True)

        mp4_path = os.path.join(output_dir, f"{self.video_cnt}.mp4")
        video_writer = imageio.get_writer(mp4_path, fps=30)
        for img in self.images:
            if img is not None:
                video_writer.append_data(img)
        video_writer.close()

        self.video_cnt += 1
        self.images = []

    def update_reset_state_ids(self):
        """Update reset state IDs (for multi-task support)."""
        pass

    def close(self):
        """Close the environment."""
        self.env.close()

    @property
    def is_start(self):
        return getattr(self, "_is_start", True)

    @is_start.setter
    def is_start(self, value):
        self._is_start = value
