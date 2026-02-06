# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf PPO agent configuration for Install Trocar task with G129 + Dex3.

This configuration is designed for VLA (Vision-Language-Action) post-training
with the GR00T model on the Install Trocar task.

NOTE: This file should NOT import any isaaclab/pxr modules, as it is loaded
by RLinf workers before AppLauncher starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class StateSpec:
    """Specification for a state tensor slice."""
    key: str
    """Key in policy observation dict."""
    slice: tuple[int, int] | None = None
    """Optional slice range [start, end]. If None, use full tensor."""


@dataclass
class GR00TStateSpec:
    """Specification for GR00T state mapping."""
    gr00t_key: str
    """GR00T state key (e.g., 'state.left_arm')."""
    slice: tuple[int, int]
    """Slice range from concatenated states."""


@dataclass
class GR00TVideoMapping:
    """Configuration for video key mapping to GR00T format."""
    main_images: str = "video.room_view"
    """GR00T key for main camera."""
    extra_view_images: list[str] = field(default_factory=lambda: [
        "video.left_wrist_view",
        "video.right_wrist_view",
    ])
    """GR00T keys for extra cameras."""


@dataclass
class GR00TMapping:
    """Configuration for IsaacLab -> GR00T format conversion."""
    video: GR00TVideoMapping = field(default_factory=GR00TVideoMapping)
    """Video key mapping."""
    state: list[GR00TStateSpec] = field(default_factory=lambda: [
        GR00TStateSpec(gr00t_key="state.left_arm", slice=(0, 7)),
        GR00TStateSpec(gr00t_key="state.right_arm", slice=(7, 14)),
        GR00TStateSpec(gr00t_key="state.left_hand", slice=(14, 21)),
        GR00TStateSpec(gr00t_key="state.right_hand", slice=(21, 28)),
    ])
    """State key mapping with slicing."""


@dataclass
class ActionMapping:
    """Configuration for GR00T -> IsaacLab action conversion."""
    prefix_pad: int = 15
    """Padding at front (e.g., for G129 body joints that aren't controlled)."""
    suffix_pad: int = 0
    """Padding at end."""


@dataclass
class RLinfObsMapCfg:
    """Configuration for observation mapping from IsaacLab to RLinf/GR00T format.
    
    This defines the complete pipeline:
    1. IsaacLab obs -> RLinf format (_wrap_obs)
    2. RLinf format -> GR00T format (_convert_isaaclab_obs_to_gr00t)
    3. GR00T action -> IsaacLab action (_convert_gr00t_to_isaaclab_action)
    """
    
    # --- IsaacLab -> RLinf mapping ---
    main_images: str = "front_camera"
    """Key for main camera image in IsaacLab obs."""
    
    extra_view_images: list[str] = field(default_factory=lambda: [
        "left_wrist_camera",
        "right_wrist_camera",
    ])
    """Keys for extra camera images to stack."""
    
    states: list[StateSpec | str] = field(default_factory=lambda: [
        StateSpec(key="robot_joint_state", slice=(15, 29)),
        StateSpec(key="robot_dex3_joint_state", slice=None),
    ])
    """State specs: key strings or StateSpec with slicing."""
    
    task_description: str = "install trocar from box"
    """Task description for language conditioning."""
    
    # --- RLinf -> GR00T mapping ---
    gr00t_mapping: GR00TMapping = field(default_factory=GR00TMapping)
    """Configuration for GR00T format conversion."""
    
    action_mapping: ActionMapping = field(default_factory=ActionMapping)
    """Configuration for action conversion."""
    
    # --- Model configuration ---
    converter_type: str = "isaaclab"
    """Converter type name for RLinf simulation_io registry."""
    
    embodiment_tag: str = "new_embodiment"
    """Embodiment tag for GR00T model."""
    
    embodiment_tag_id: int = 31
    """Numeric ID for the embodiment tag."""
    
    data_config_class: str = "policy.gr00t_config:IsaacLabDataConfig"
    """Module path to GR00T data config class."""
    
    def to_json_dict(self) -> dict:
        """Convert to JSON-serializable dict for env var passing."""
        from dataclasses import asdict
        
        result = {
            "main_images": self.main_images,
            "extra_view_images": self.extra_view_images,
            "states": [],
            "gr00t_mapping": {
                "video": {
                    "main_images": self.gr00t_mapping.video.main_images,
                    "extra_view_images": self.gr00t_mapping.video.extra_view_images,
                },
                "state": [
                    {"gr00t_key": s.gr00t_key, "slice": list(s.slice)}
                    for s in self.gr00t_mapping.state
                ],
            },
            "action_mapping": {
                "prefix_pad": self.action_mapping.prefix_pad,
                "suffix_pad": self.action_mapping.suffix_pad,
            },
        }
        
        for spec in self.states:
            if isinstance(spec, str):
                result["states"].append(spec)
            else:
                d = {"key": spec.key}
                if spec.slice is not None:
                    d["slice"] = list(spec.slice)
                result["states"].append(d)
        
        return result


@dataclass
class RLinfAlgorithmCfg:
    """Configuration for RLinf algorithm parameters."""
    
    loss_type: Literal["actor_critic", "embodied_sac", "grpo"] = "actor_critic"
    adv_type: Literal["gae", "grpo", "reinpp_baseline"] = "gae"
    group_size: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio_high: float = 0.2
    clip_ratio_low: float = 0.2
    value_clip: float = 0.2
    entropy_bonus: float = 0.0
    kl_beta: float = 0.0
    normalize_advantages: bool = True
    reward_type: Literal["step_level", "chunk_level"] = "chunk_level"
    logprob_type: Literal["step_level", "chunk_level"] = "chunk_level"
    update_epoch: int = 4
    rollout_epoch: int = 8


@dataclass
class RLinfModelCfg:
    """Configuration for RLinf model."""
    
    model_path: str = ""
    """Path to the pretrained model checkpoint."""
    
    model_type: str = "gr00t"
    """Type of model."""
    
    precision: str = "bf16"
    """Model precision."""
    
    add_value_head: bool = True
    """Whether to add a value head."""
    
    num_action_chunks: int = 1
    """Number of action chunks."""
    
    action_dim: int = 28
    """Action dimension (14 shoulder + 14 hand joints)."""
    
    policy_setup: str = "g129_dex3"
    """Policy setup identifier."""
    
    embodiment_tag: str = "g129_dex3"
    """Embodiment tag."""


@dataclass
class RLinfOptimCfg:
    """Configuration for optimizer."""
    
    lr: float = 5e-6
    value_lr: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    clip_grad: float = 1.0


@dataclass
class RLinfLoggerCfg:
    """Configuration for logging."""
    
    log_path: str = "logs/rlinf"
    project_name: str = "rlinf"
    experiment_name: str = "install_trocar_gr00t"
    logger_backends: list[str] = field(default_factory=lambda: ["tensorboard"])


@dataclass
class RLinfRunnerCfg:
    """Configuration for RLinf runner."""
    
    task_type: str = "embodied"
    max_epochs: int = 1000
    max_steps: int = -1
    val_check_interval: int = -1
    save_interval: int = 2
    only_eval: bool = False
    resume_dir: str | None = None
    ckpt_path: str | None = None


@dataclass
class RLinfEnvCfg:
    """Configuration for environment."""
    
    total_num_envs: int = 56
    max_episode_steps: int = 256
    max_steps_per_rollout_epoch: int = 256
    auto_reset: bool = False
    ignore_terminations: bool = False
    video_save: bool = False


@dataclass
class RLinfClusterCfg:
    """Configuration for distributed cluster."""
    
    num_nodes: int = 1
    component_placement: dict = field(default_factory=lambda: {"actor,env,rollout": "0-3, 5-7"})


@dataclass
class RLinfInstallTrocarPPOCfg:
    """Complete RLinf PPO configuration for Install Trocar task.
    
    This configuration is used as the `rlinf_cfg_entry_point` for the task
    registration, enabling IsaacLab users to train this task with RLinf.
    """
    
    seed: int = 1234
    
    algorithm: RLinfAlgorithmCfg = field(default_factory=RLinfAlgorithmCfg)
    model: RLinfModelCfg = field(default_factory=RLinfModelCfg)
    optim: RLinfOptimCfg = field(default_factory=RLinfOptimCfg)
    runner: RLinfRunnerCfg = field(default_factory=RLinfRunnerCfg)
    logger: RLinfLoggerCfg = field(default_factory=RLinfLoggerCfg)
    env: RLinfEnvCfg = field(default_factory=RLinfEnvCfg)
    cluster: RLinfClusterCfg = field(default_factory=RLinfClusterCfg)
    obs_map: RLinfObsMapCfg = field(default_factory=RLinfObsMapCfg)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
