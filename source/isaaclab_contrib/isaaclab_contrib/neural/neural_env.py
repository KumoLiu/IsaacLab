# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Model-independent, single-environment neural backend for native Isaac Lab.

Concrete PhysicsManagers implement four model lifecycle hooks, without another
Gym environment. They own inference and task reward; Isaac Lab owns episode resets.
A policy is simply a callable from batched observations to batched actions.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import MISSING

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.physics import PhysicsCfg, PhysicsEvent, PhysicsManager
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

NeuralPolicy = Callable[[dict[str, torch.Tensor]], torch.Tensor]


class _EmptySceneData(SceneDataBackend):
    """Publish zero transforms: neural observations are not rigid-body poses."""

    def __init__(self, device: str):
        self._transforms = SceneDataFormat.Transform()
        self._transforms.transforms = wp.empty(0, dtype=wp.transformf, device=device)

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        return self._transforms

    @property
    def transform_count(self) -> int:
        return 0

    @property
    def transform_paths(self) -> list[str]:
        return []


class NeuralPhysicsManager(PhysicsManager):
    """Run model lifecycle hooks only through native simulation callbacks."""

    _model_loaded: bool = False
    _pending_action: np.ndarray | None = None
    observation: dict[str, torch.Tensor] | None = None
    transition: dict | None = None

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        super().initialize(sim_context)
        cls._model_loaded = False
        cls._pending_action = None
        cls.observation = None
        cls.transition = None
        cls._scene_data = _EmptySceneData(sim_context.cfg.device)

    @classmethod
    def validate_env_cfg(cls, cfg: NeuralDirectEnvCfg) -> None:
        """Check declared model spaces and simulated step duration [s] before startup."""
        physics = cfg.sim.physics
        if not math.isclose(physics.step_dt, cfg.sim.dt, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Backend step_dt must match sim.dt [s]")
        if cfg.action_space != physics.action_space or cfg.observation_space != physics.observation_space:
            raise ValueError("Configured spaces must match the backend spaces")

    @classmethod
    def _load_model(cls) -> None:
        """Load model resources; failed startup must also be safe to close."""
        raise NotImplementedError

    @classmethod
    def _reset_model(cls, *, seed: int, options: dict) -> tuple[dict[str, np.ndarray], dict]:
        """Return initial unbatched observations and metadata."""
        raise NotImplementedError

    @classmethod
    def _step_model(cls, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Return observation, reward, terminated, truncated, info; never auto-reset."""
        raise NotImplementedError

    @classmethod
    def _close_model(cls) -> None:
        """Release model resources, including partially initialized resources."""
        raise NotImplementedError

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """Create inference resources once; this is not an episode reset."""
        if not cls._model_loaded:
            cls._load_model()
            cls._model_loaded = True
        if not soft:
            cls.dispatch_event(PhysicsEvent.MODEL_INIT)
        cls.dispatch_event(PhysicsEvent.PHYSICS_READY)

    @classmethod
    def reset_episode(cls, *, seed: int, options: dict) -> dict:
        """Reset model state; simulation-wide time remains monotonic."""
        cls._pending_action = None
        cls.transition = None
        cls.observation = None
        observation, info = cls._reset_model(seed=seed, options=copy.deepcopy(options))
        cls._store_observation(observation)
        return copy.deepcopy(info)

    @classmethod
    def _store_observation(cls, observation: dict[str, np.ndarray]) -> None:
        if not PhysicsManager._cfg.observation_space.contains(observation):
            raise ValueError("Backend observation does not match its declared space")
        cls.observation = {
            key: torch.from_numpy(value.copy()).unsqueeze(0).to(cls.get_device()) for key, value in observation.items()
        }

    @classmethod
    def set_action(cls, action: torch.Tensor) -> None:
        """Queue one batched action; units and shape are defined by the adapter."""
        if cls.observation is None:
            raise RuntimeError("Reset the environment before applying actions")
        space = PhysicsManager._cfg.action_space
        if action.shape != (1, *space.shape) or not torch.isfinite(action).all():
            raise ValueError(f"Expected finite actions with shape {(1, *space.shape)}")
        array = action[0].detach().cpu().numpy().astype(space.dtype, copy=True)
        if not space.contains(array):
            raise ValueError("Action is outside the backend action space")
        if cls._pending_action is not None:
            raise RuntimeError("An action is already queued")
        cls._pending_action = array

    @classmethod
    def step(cls) -> None:
        """Execute exactly one backend transition and retain its terminal data."""
        if cls._pending_action is None:
            raise RuntimeError("No action queued for the neural backend")
        action = cls._pending_action
        cls._pending_action = None
        cls.observation = None
        cls.transition = None
        observation, reward, terminated, truncated, info = cls._step_model(action)
        if np.ndim(reward) != 0 or not np.isfinite(reward):
            raise ValueError("Backend reward must be a finite scalar")
        cls._store_observation(observation)
        cls.transition = {
            **copy.deepcopy(info),
            **{key: value.copy() for key, value in observation.items()},
            "reward": np.asarray(float(reward)),
            "terminated": np.asarray(bool(terminated)),
            "truncated": np.asarray(bool(truncated)),
        }
        PhysicsManager._sim_time += cls.get_physics_dt()

    @classmethod
    def forward(cls) -> None:
        """No-op: rendering/forward must not advance the model."""

    @classmethod
    def render(cls) -> np.ndarray | None:
        """Optionally return current RGB pixels without advancing the model."""
        return None

    @classmethod
    def set_decimation(cls, decimation: int) -> None:
        if decimation != 1:
            raise ValueError("Neural backends require decimation=1")

    @classmethod
    def get_scene_data_backend(cls) -> SceneDataBackend:
        return cls._scene_data

    @classmethod
    def close(cls) -> None:
        try:
            super().close()
        finally:
            try:
                cls._close_model()
            finally:
                cls._model_loaded = False
                cls._pending_action = None
                cls.observation = None
                cls.transition = None
                cls._scene_data = None


@configclass
class NeuralPhysicsCfg(PhysicsCfg):
    """Select a concrete manager and declare its model input/output contract."""

    class_type: type[PhysicsManager] = NeuralPhysicsManager
    action_space: gym.spaces.Box = MISSING
    observation_space: gym.spaces.Dict = MISSING
    step_dt: float = MISSING


@configclass
class NeuralDirectEnvCfg(DirectRLEnvCfg):
    """Declare action/observation spaces, step duration and task horizon [s]."""

    sim: SimulationCfg = SimulationCfg(
        device="cpu", dt=1.0, physics=NeuralPhysicsCfg(), create_stage_in_memory=True, visualizer_cfgs=[]
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=1.0, replicate_physics=False, filter_collisions=False
    )
    decimation: int = 1
    seed: int = 1234
    compute_final_obs: bool = True
    ui_window_class_type: type | str | None = None
    reset_options: dict = {}


class NeuralDirectEnv(DirectRLEnv):
    """Model-independent hooks using native step/close and delegating core reset.

    Observations are batched tensors under ``obs['policy']``. Terminal observations
    are preserved in ``extras['final_obs']`` before native Same-Step autoreset.
    """

    cfg: NeuralDirectEnvCfg
    metadata = dict(DirectRLEnv.metadata)

    def __init__(self, cfg: NeuralDirectEnvCfg, render_mode: str | None = None, **kwargs):
        self._is_closed = True
        cfg.validate()
        has_assets = any(
            name not in InteractiveSceneCfg.__dataclass_fields__ and value is not None
            for name, value in vars(cfg.scene).items()
        )
        if type(cfg.scene) is not InteractiveSceneCfg or cfg.scene.num_envs != 1 or has_assets:
            raise ValueError("Neural backends currently support one empty InteractiveScene, without assets")
        if not issubclass(cfg.sim.physics.class_type, NeuralPhysicsManager):
            raise ValueError("Select a NeuralPhysicsManager backend")
        if cfg.sim.device != "cpu" or cfg.decimation != 1:
            raise ValueError("The current neural frontend requires CPU tensors and decimation=1")
        if not math.isfinite(cfg.sim.dt) or cfg.sim.dt <= 0:
            raise ValueError("sim.dt must be positive and finite [s]")
        if not math.isfinite(cfg.episode_length_s) or cfg.episode_length_s <= 0:
            raise ValueError("episode_length_s must be positive and finite [s]")
        if not isinstance(cfg.action_space, gym.spaces.Box) or cfg.action_space.dtype.kind != "f":
            raise ValueError("Use a floating-point Box action space")
        if not isinstance(cfg.observation_space, gym.spaces.Dict) or not all(
            isinstance(space, gym.spaces.Box) for space in cfg.observation_space.values()
        ):
            raise ValueError("Use a flat Dict of Box observation spaces")
        if cfg.sim.visualizer_cfgs or cfg.events or cfg.video_recorders:
            raise ValueError("Scene visualizers/events/recorders are unsupported; use generated RGB rendering")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only generated rgb_array rendering is supported")
        cfg.sim.physics.class_type.validate_env_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Forward model-specific reset options, then use the native reset lifecycle."""
        if options is not None:
            self.cfg.reset_options = copy.deepcopy(options)
        return super().reset(seed=seed, options=options)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.extras = {}

    def _apply_action(self) -> None:
        self.sim.physics_manager.set_action(self.actions)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observation = self.sim.physics_manager.observation
        if observation is None:
            raise RuntimeError("Call reset() before requesting observations")
        return {"policy": {key: value.clone() for key, value in observation.items()}}

    def _get_rewards(self) -> torch.Tensor:
        transition = self.sim.physics_manager.transition
        self.extras["neural_transition"] = copy.deepcopy(transition)
        return torch.tensor([float(transition["reward"])], device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        transition = self.sim.physics_manager.transition
        terminated = torch.tensor([bool(transition["terminated"])], device=self.device)
        truncated = torch.tensor([bool(transition["truncated"])], device=self.device)
        return terminated, truncated | (self.episode_length_buf >= self.max_episode_length)

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        if len(env_ids) != 1 or int(env_ids[0]) != 0:
            raise ValueError("Only environment index 0 is supported")
        super()._reset_idx(env_ids)
        self.extras["reset_info"] = self.sim.physics_manager.reset_episode(
            seed=int(torch.initial_seed()), options=self.cfg.reset_options
        )

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Read adapter pixels without invoking a model or a USD renderer."""
        if self.render_mode is None:
            return None
        image = self.sim.physics_manager.render()
        return None if image is None else image.copy()
