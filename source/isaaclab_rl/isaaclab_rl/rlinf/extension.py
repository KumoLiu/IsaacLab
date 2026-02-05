# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf extension module for IsaacLab tasks.

This module is loaded by RLinf's Worker._load_user_extensions() when
RLINF_EXT_MODULE=isaaclab_rl.rlinf.extension is set in the environment.

It registers IsaacLab tasks into RLinf's registries, allowing IsaacLab users
to train on their tasks without modifying RLinf source code.

Usage:
    # Set the extension module and task to register
    export RLINF_EXT_MODULE=isaaclab_rl.rlinf.extension
    export RLINF_ISAACLAB_TASKS="Isaac-Install-Trocar-G129-Dex3-RLinf-v0"
    
    # For multiple tasks, separate with comma:
    export RLINF_ISAACLAB_TASKS="Isaac-Task1-v0,Isaac-Task2-v0"
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_registered = False

# Module-level config cache (shared across all dynamically created classes)
_shared_obs_map: dict | None = None
_shared_task_description = ""


def register() -> None:
    """Register IsaacLab extensions into RLinf.

    This function is called automatically by RLinf's Worker._load_user_extensions()
    when RLINF_EXT_MODULE=isaaclab_rl.rlinf.extension is set.

    It performs the following registrations:
    1. Registers GR00T obs/action converters
    2. Registers GR00T data config (if RLINF_DATA_CONFIG is set)
    3. Patches GR00T get_model for custom embodiment (if RLINF_EMBODIMENT_TAG is set)
    4. Registers task IDs from RLINF_ISAACLAB_TASKS env var into REGISTER_ISAACLAB_ENVS
    """
    global _registered
    if _registered:
        return
    _registered = True

    logger.info("isaaclab_rl.rlinf.extension: Registering IsaacLab extensions...")

    _register_gr00t_converters()
    _register_gr00t_data_config()
    _patch_gr00t_get_model()
    _register_isaaclab_envs()

    logger.info("isaaclab_rl.rlinf.extension: Registration complete.")


def _register_gr00t_data_config() -> None:
    """Register GR00T data config for custom embodiment.
    
    Set RLINF_DATA_CONFIG to specify the data config module path.
    Example: export RLINF_DATA_CONFIG="policy.gr00t_config"
    
    This imports the module which should add to DATA_CONFIG_MAP.
    """
    data_config_module = os.environ.get("RLINF_DATA_CONFIG", "")
    if not data_config_module:
        logger.debug("RLINF_DATA_CONFIG not set, skipping data config registration")
        return
    
    try:
        import importlib
        importlib.import_module(data_config_module)
        logger.info(f"Registered GR00T data config from: {data_config_module}")
    except ImportError as e:
        logger.warning(f"Failed to import data config module '{data_config_module}': {e}")


def _patch_embodiment_tags() -> None:
    """Add custom embodiment tag to RLinf's EmbodimentTag enum and mapping.
    
    Set RLINF_EMBODIMENT_TAG to specify the embodiment tag name.
    Set RLINF_EMBODIMENT_TAG_ID to specify the tag ID (default: 31).
    
    Example:
        export RLINF_EMBODIMENT_TAG="new_embodiment"
        export RLINF_EMBODIMENT_TAG_ID="31"
    """
    from rlinf.models.embodiment.gr00t import embodiment_tags
    
    embodiment_tag = os.environ.get("RLINF_EMBODIMENT_TAG", "")
    if not embodiment_tag:
        return
    
    tag_id = int(os.environ.get("RLINF_EMBODIMENT_TAG_ID", "31"))
    
    # Add to enum if not exists
    tag_upper = embodiment_tag.upper().replace("-", "_")
    if not hasattr(embodiment_tags.EmbodimentTag, tag_upper):
        from enum import Enum
        
        existing_members = {e.name: e.value for e in embodiment_tags.EmbodimentTag}
        existing_members[tag_upper] = embodiment_tag
        NewEmbodimentTag = Enum("EmbodimentTag", existing_members)
        
        embodiment_tags.EmbodimentTag = NewEmbodimentTag
        logger.info(f"Added EmbodimentTag.{tag_upper} = '{embodiment_tag}'")
    
    # Add to mapping if not exists
    if embodiment_tag not in embodiment_tags.EMBODIMENT_TAG_MAPPING:
        embodiment_tags.EMBODIMENT_TAG_MAPPING[embodiment_tag] = tag_id
        logger.info(f"Added EMBODIMENT_TAG_MAPPING['{embodiment_tag}'] = {tag_id}")


def _patch_gr00t_get_model() -> None:
    """Monkeypatch RLinf's GR00T get_model to support custom embodiment.
    
    Required environment variables:
        RLINF_EMBODIMENT_TAG: The embodiment tag name (e.g., "new_embodiment")
        RLINF_DATA_CONFIG_CLASS: Full path to data config class 
                                 (e.g., "policy.gr00t_config:UnitreeG1SimDataConfig")
    
    If RLINF_EMBODIMENT_TAG is not set, this function does nothing.
    """
    embodiment_tag = os.environ.get("RLINF_EMBODIMENT_TAG", "")
    if not embodiment_tag:
        logger.debug("RLINF_EMBODIMENT_TAG not set, skipping get_model patch")
        return
    
    # First patch the embodiment tags
    _patch_embodiment_tags()
    
    import rlinf.models.embodiment.gr00t as rlinf_gr00t_mod
    
    original_get_model = rlinf_gr00t_mod.get_model
    _embodiment_tag = embodiment_tag  # Capture for closure
    
    def patched_get_model(cfg, torch_dtype=None):
        import torch
        
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        
        # If not our custom embodiment, use original logic
        if cfg.embodiment_tag != _embodiment_tag:
            return original_get_model(cfg, torch_dtype=torch_dtype)
        
        # Handle custom embodiment
        from pathlib import Path
        
        from gr00t.experiment.data_config import load_data_config
        from rlinf.models.embodiment.gr00t.gr00t_action_model import GR00T_N1_5_ForRLActionPrediction
        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity
        from rlinf.utils.patcher import Patcher
        
        # Apply RLinf's standard EmbodimentTag patches
        Patcher.clear()
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EmbodimentTag",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
        )
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        )
        Patcher.apply()
        
        # Load data config
        data_config_class = os.environ.get("RLINF_DATA_CONFIG_CLASS", "")
        if not data_config_class:
            raise ValueError(
                f"RLINF_DATA_CONFIG_CLASS must be set for embodiment_tag='{_embodiment_tag}'. "
                f"Example: export RLINF_DATA_CONFIG_CLASS='policy.gr00t_config:UnitreeG1SimDataConfig'"
            )
        
        data_config = load_data_config(data_config_class)
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        
        model_path = Path(cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            embodiment_tag=cfg.embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=cfg.denoising_steps,
            output_action_chunks=cfg.num_action_chunks,
            obs_converter_type=cfg.obs_converter_type,
            tune_visual=False,
            tune_llm=False,
            rl_head_config=cfg.rl_head_config,
        )
        model.to(torch_dtype)
        
        if cfg.rl_head_config.add_value_head:
            model.action_head.value_head._init_weights()
        if cfg.rl_head_config.disable_dropout:
            replace_dropout_with_identity(model)
        
        logger.info(f"Loaded GR00T model with embodiment_tag='{cfg.embodiment_tag}'")
        return model
    
    rlinf_gr00t_mod.get_model = patched_get_model
    logger.info(f"Patched get_model for embodiment_tag='{embodiment_tag}'")


def _register_gr00t_converters() -> None:
    """Register GR00T obs/action converters for IsaacLab tasks.
    
    This registers a generic "isaaclab" converter that uses the gr00t_mapping
    from RLINF_OBS_MAP_JSON to convert observations and actions.
    """
    from rlinf.models.embodiment.gr00t import simulation_io

    # Register the generic IsaacLab converter
    converter_type = os.environ.get("RLINF_CONVERTER_TYPE", "isaaclab")
    
    if converter_type not in simulation_io.OBS_CONVERSION:
        simulation_io.OBS_CONVERSION[converter_type] = _convert_isaaclab_obs_to_gr00t
        logger.info(f"Registered obs converter: {converter_type}")
    
    if converter_type not in simulation_io.ACTION_CONVERSION:
        simulation_io.ACTION_CONVERSION[converter_type] = _convert_gr00t_to_isaaclab_action
        logger.info(f"Registered action converter: {converter_type}")


def _convert_isaaclab_obs_to_gr00t(env_obs: dict) -> dict:
    """Convert IsaacLab env observations to GR00T format.
    
    Uses gr00t_mapping from RLINF_OBS_MAP_JSON to configure the conversion.
    
    Expected input (from _wrap_obs):
      - main_images: (B, H, W, C) torch tensor
      - extra_view_images: (B, N, H, W, C) torch tensor
      - states: (B, D) torch tensor
      - task_descriptions: list[str]
    
    gr00t_mapping config example:
      {
        "video": {
          "main_images": "video.room_view",
          "extra_view_images": ["video.left_wrist_view", "video.right_wrist_view"]
        },
        "state": [
          {"gr00t_key": "state.left_arm", "slice": [0, 7]},
          {"gr00t_key": "state.right_arm", "slice": [7, 14]},
          {"gr00t_key": "state.left_hand", "slice": [14, 21]},
          {"gr00t_key": "state.right_hand", "slice": [21, 28]}
        ]
      }
    """
    import json
    import torch
    
    groot_obs = {}
    
    # Load mapping config
    obs_map_json = os.environ.get("RLINF_OBS_MAP_JSON", "{}")
    try:
        obs_map = json.loads(obs_map_json)
    except json.JSONDecodeError:
        obs_map = {}
    
    gr00t_mapping = obs_map.get("gr00t_mapping", {})
    video_mapping = gr00t_mapping.get("video", {})
    state_mapping = gr00t_mapping.get("state", [])
    
    # Convert main_images -> video.xxx
    if "main_images" in env_obs:
        main = env_obs["main_images"]
        gr00t_key = video_mapping.get("main_images", "video.room_view")
        if isinstance(main, torch.Tensor):
            # (B, H, W, C) -> (B, T=1, H, W, C)
            groot_obs[gr00t_key] = main.unsqueeze(1).cpu().numpy()
    
    # Convert extra_view_images -> video.xxx
    if "extra_view_images" in env_obs:
        extra = env_obs["extra_view_images"]  # (B, N, H, W, C)
        extra_keys = video_mapping.get("extra_view_images", [])
        if isinstance(extra, torch.Tensor):
            for i, key in enumerate(extra_keys):
                if i < extra.shape[1]:
                    # (B, H, W, C) -> (B, T=1, H, W, C)
                    groot_obs[key] = extra[:, i].unsqueeze(1).cpu().numpy()
    
    # Convert states -> state.xxx with slicing
    if "states" in env_obs and state_mapping:
        states = env_obs["states"]  # (B, D)
        if isinstance(states, torch.Tensor):
            states_np = states.unsqueeze(1).cpu().numpy()  # (B, T=1, D)
            for spec in state_mapping:
                gr00t_key = spec.get("gr00t_key")
                slice_range = spec.get("slice", [0, states_np.shape[-1]])
                if gr00t_key:
                    groot_obs[gr00t_key] = states_np[:, :, slice_range[0]:slice_range[1]]
    
    # Pass through task descriptions
    groot_obs["annotation.human.action.task_description"] = env_obs.get("task_descriptions", [])
    
    return groot_obs


def _convert_gr00t_to_isaaclab_action(action_chunk: dict, chunk_size: int = 1) -> Any:
    """Convert GR00T action output to IsaacLab env action format.
    
    Uses action_mapping from RLINF_OBS_MAP_JSON to configure the conversion.
    
    action_mapping config example:
      {
        "prefix_pad": 15,  # Pad zeros at front (for G129 body joints)
        "suffix_pad": 0    # Pad zeros at end
      }
    """
    import json
    import numpy as np
    
    # Load mapping config
    obs_map_json = os.environ.get("RLINF_OBS_MAP_JSON", "{}")
    try:
        obs_map = json.loads(obs_map_json)
    except json.JSONDecodeError:
        obs_map = {}
    
    action_mapping = obs_map.get("action_mapping", {})
    prefix_pad = action_mapping.get("prefix_pad", 0)
    suffix_pad = action_mapping.get("suffix_pad", 0)
    
    # Concatenate all action parts
    action_parts = [v[:, :chunk_size, :] for v in action_chunk.values()]
    action_concat = np.concatenate(action_parts, axis=-1)
    
    # Apply padding
    if prefix_pad > 0 or suffix_pad > 0:
        action_concat = np.pad(
            action_concat,
            ((0, 0), (0, 0), (prefix_pad, suffix_pad)),
            mode="constant",
            constant_values=0,
        )
    
    return action_concat


def _register_isaaclab_envs() -> None:
    """Register IsaacLab tasks into RLinf's REGISTER_ISAACLAB_ENVS map."""
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    # Get task IDs to register from environment variable
    tasks_env = os.environ.get("RLINF_ISAACLAB_TASKS", "")
    if not tasks_env:
        logger.warning("RLINF_ISAACLAB_TASKS not set, no tasks to register")
        logger.info("Set RLINF_ISAACLAB_TASKS='Isaac-MyTask-v0' to register tasks")
        return

    task_ids = [t.strip() for t in tasks_env.split(",") if t.strip()]
    logger.info(f"Tasks to register: {task_ids}")

    for task_id in task_ids:
        if task_id in REGISTER_ISAACLAB_ENVS:
            logger.debug(f"Task '{task_id}' already registered, skipping")
            continue

        # Create a generic wrapper class for this task
        env_class = _create_generic_env_wrapper(task_id)
        REGISTER_ISAACLAB_ENVS[task_id] = env_class
        logger.info(f"Registered IsaacLab task '{task_id}' for RLinf")

    logger.debug(f"REGISTER_ISAACLAB_ENVS now contains: {list(REGISTER_ISAACLAB_ENVS.keys())}")


def _create_generic_env_wrapper(task_id: str) -> type:
    """Create a generic wrapper class for an IsaacLab task.

    The wrapper class will load the task configuration at runtime
    (after AppLauncher starts) and configure observation mapping accordingly.

    This follows the same pattern as i4h's rlinf_ext: all isaaclab-dependent
    imports happen inside _make_env_function, after AppLauncher starts.

    Args:
        task_id: The gymnasium task ID.

    Returns:
        A class that inherits from IsaaclabBaseEnv.
    """
    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

    _task_id = task_id

    class IsaacLabGenericEnv(IsaaclabBaseEnv):
        """Generic environment wrapper for IsaacLab tasks.
        
        This wrapper loads the task configuration at runtime and applies 
        observation mapping based on rlinf_cfg_entry_point.
        
        Key design:
        - Config (obs_map) is loaded in __init__ (parent process) because _wrap_obs runs in parent
        - IsaacLab env is created in _make_env_function (child process via SubProcIsaacLabEnv)
        
        Note: obs_map is stored in module-level variables (_shared_obs_map) to be
        shared across all dynamically created IsaacLabGenericEnv classes.
        """

        def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
            # Load obs_map BEFORE calling super().__init__ because:
            # 1. super().__init__ creates SubProcIsaacLabEnv which runs in child process
            # 2. _wrap_obs is called in THIS (parent) process, so it needs obs_map here
            self._load_obs_map_from_env()
            super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        def _load_obs_map_from_env(self) -> None:
            """Load obs_map from JSON environment variable.
            
            Users must set RLINF_OBS_MAP_JSON in their run script to configure
            observation mapping for their task. Example:
            
                export RLINF_OBS_MAP_JSON='{
                    "main_images": "front_camera",
                    "wrist_images": "wrist_camera",
                    "state_keys": ["joint_pos", "joint_vel"],
                    "convert_quat_to_axisangle": false
                }'
            """
            import json
            import isaaclab_rl.rlinf.extension as ext_module

            if ext_module._shared_obs_map is not None:
                return  # Already loaded

            obs_map_json = os.environ.get("RLINF_OBS_MAP_JSON", "")
            task_desc = os.environ.get("RLINF_TASK_DESCRIPTION", "")
            
            if not obs_map_json:
                logger.error(
                    f"RLINF_OBS_MAP_JSON not set for task '{_task_id}'!\n"
                    f"Please set this environment variable in your run script. Example:\n"
                    f'  export RLINF_OBS_MAP_JSON=\'{{"main_images":"camera","state_keys":["joint_pos"]}}\''
                )
                # Use empty obs_map - _wrap_obs will return minimal observation
                return

            try:
                ext_module._shared_obs_map = json.loads(obs_map_json)
                ext_module._shared_task_description = task_desc
                logger.info(f"Loaded obs_map: {ext_module._shared_obs_map}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in RLINF_OBS_MAP_JSON: {e}")

        def _make_env_function(self):
            """Create the environment factory function.
            
            This function runs in child process (via SubProcIsaacLabEnv).
            All isaaclab-dependent imports happen here, after AppLauncher starts.
            """
            def make_env_isaaclab():
                from isaaclab.app import AppLauncher

                sim_app = AppLauncher(headless=True, enable_cameras=True).app
                import gymnasium as gym
                from isaaclab_tasks.utils import load_cfg_from_registry

                isaac_env_cfg = load_cfg_from_registry(self.isaaclab_env_id, "env_cfg_entry_point")
                isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs

                env = gym.make(self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array").unwrapped
                return env, sim_app

            return make_env_isaaclab

        def _wrap_obs(self, obs):
            """Convert observations to RLinf format.
            
            Output format matches i4h's convention:
              - main_images: (B, H, W, C) - single main camera
              - extra_view_images: (B, N, H, W, C) - stacked extra cameras
              - states: (B, D) - concatenated state vector
              - task_descriptions: list[str] - task descriptions
            
            obs_map JSON config format:
              {
                "main_images": "front_camera",
                "extra_view_images": ["left_wrist_camera", "right_wrist_camera"],
                "states": [
                  {"key": "robot_joint_state", "slice": [15, 29]},
                  {"key": "robot_dex3_joint_state"}
                ]
              }
            """
            import torch
            import isaaclab_rl.rlinf.extension as ext_module

            policy_obs = obs.get("policy", obs)
            camera_obs = obs.get("camera_images", {})

            rlinf_obs = {
                "task_descriptions": [self.task_description] * self.num_envs,
            }

            obs_map = ext_module._shared_obs_map
            if not obs_map:
                logger.warning("obs_map not loaded, returning minimal observation")
                return rlinf_obs

            # main_images: single camera key -> (B, H, W, C)
            main_key = obs_map.get("main_images")
            if main_key and main_key in camera_obs:
                rlinf_obs["main_images"] = camera_obs[main_key]

            # extra_view_images: camera key(s) -> stack to (B, N, H, W, C)
            extra_keys = obs_map.get("extra_view_images")
            if extra_keys:
                if isinstance(extra_keys, str):
                    extra_keys = [extra_keys]
                extra_imgs = [camera_obs[k] for k in extra_keys if k in camera_obs]
                if extra_imgs:
                    rlinf_obs["extra_view_images"] = torch.stack(extra_imgs, dim=1)

            # states: list of state specs -> concatenate to (B, D)
            # Each spec: string "key" or dict {"key": "...", "slice": [start, end]}
            state_specs = obs_map.get("states")
            if state_specs:
                state_parts = []
                for spec in state_specs:
                    if isinstance(spec, str):
                        state = policy_obs.get(spec)
                        if state is not None:
                            state_parts.append(state)
                    elif isinstance(spec, dict):
                        state = policy_obs.get(spec.get("key"))
                        if state is not None:
                            slice_range = spec.get("slice")
                            if slice_range:
                                state = state[:, slice_range[0]:slice_range[1]]
                            state_parts.append(state)
                if state_parts:
                    rlinf_obs["states"] = torch.cat(state_parts, dim=-1)

            return rlinf_obs

        def add_image(self, obs):
            """Get image for video logging."""
            import isaaclab_rl.rlinf.extension as ext_module
            camera_obs = obs.get("camera_images", {})
            obs_map = ext_module._shared_obs_map
            
            # Try main_images key, fallback to first available camera
            main_key = obs_map.get("main_images") if obs_map else None
            if main_key and main_key in camera_obs:
                return camera_obs[main_key][0].cpu().numpy()
            
            for img in camera_obs.values():
                return img[0].cpu().numpy()
            return None

    return IsaacLabGenericEnv


