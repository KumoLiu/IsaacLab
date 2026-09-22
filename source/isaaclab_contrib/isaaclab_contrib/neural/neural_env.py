# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Model-independent, batched neural backend for native Isaac Lab.

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
        cls._ready = np.zeros(sim_context.cfg.physics.num_envs, dtype=bool)
        cls._scene_data = _EmptySceneData(sim_context.cfg.device)

    @classmethod
    def validate_env_cfg(cls, cfg: NeuralDirectEnvCfg) -> None:
        """Check declared model spaces and simulated step duration [s] before startup."""
        physics = cfg.sim.physics
        if type(physics.num_envs) is not int or physics.num_envs < 1 or physics.num_envs != cfg.scene.num_envs:
            raise ValueError("Backend num_envs must be positive and match scene.num_envs")
        if not math.isclose(physics.step_dt, cfg.sim.dt, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Backend step_dt must match sim.dt [s]")
        if cfg.action_space != physics.action_space or cfg.observation_space != physics.observation_space:
            raise ValueError("Configured spaces must match the backend spaces")

    @classmethod
    def _load_model(cls) -> None:
        """Load model resources; failed startup must also be safe to close."""
        raise NotImplementedError

    @classmethod
    def _reset_model(cls, *, env_ids: np.ndarray, seed: int, options: dict) -> tuple[dict[str, np.ndarray], dict]:
        """Reset only selected rows; return observations batched in env_ids order."""
        raise NotImplementedError

    @classmethod
    def _step_model(cls, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
        """Advance all rows; return batched observations and (N,) reward/done arrays."""
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
    def reset_episode(cls, *, seed: int, options: dict, env_ids: np.ndarray | None = None) -> dict:
        """Reset selected model states without changing other rows or simulation time."""
        if env_ids is None:
            env_ids = np.arange(PhysicsManager._cfg.num_envs)
        env_ids = np.asarray(env_ids)
        if (
            env_ids.ndim != 1
            or env_ids.dtype.kind not in "iu"
            or not len(env_ids)
            or len(np.unique(env_ids)) != len(env_ids)
            or np.any(env_ids < 0)
            or np.any(env_ids >= PhysicsManager._cfg.num_envs)
        ):
            raise ValueError("env_ids must contain distinct valid environment indices")
        cls._pending_action = None
        cls.transition = None
        cls._ready[env_ids] = False
        observation, info = cls._reset_model(env_ids=env_ids, seed=seed, options=copy.deepcopy(options))
        cls._store_observation(observation, env_ids)
        cls._ready[env_ids] = True
        return copy.deepcopy(info)

    @classmethod
    def _store_observation(cls, observation: dict[str, np.ndarray], env_ids: np.ndarray | None = None) -> None:
        count = PhysicsManager._cfg.num_envs if env_ids is None else len(env_ids)
        if not gym.vector.utils.batch_space(PhysicsManager._cfg.observation_space, count).contains(observation):
            raise ValueError("Backend observation does not match its declared space")
        if cls.observation is None:
            cls.observation = {
                key: torch.zeros(
                    (PhysicsManager._cfg.num_envs, *value.shape[1:]),
                    dtype=torch.from_numpy(value).dtype,
                    device=cls.get_device(),
                )
                for key, value in observation.items()
            }
        indices = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=cls.get_device())
        for key, value in observation.items():
            cls.observation[key][indices] = torch.from_numpy(value.copy()).to(cls.get_device())

    @classmethod
    def set_action(cls, action: torch.Tensor) -> None:
        """Queue one batched action; units and shape are defined by the adapter."""
        if cls.observation is None or not cls._ready.all():
            raise RuntimeError("Reset the environment before applying actions")
        space = gym.vector.utils.batch_space(PhysicsManager._cfg.action_space, PhysicsManager._cfg.num_envs)
        if action.shape != space.shape or not torch.isfinite(action).all():
            raise ValueError(f"Expected finite actions with shape {space.shape}")
        array = action.detach().cpu().numpy().astype(space.dtype, copy=True)
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
        cls._ready[:] = False
        observation, reward, terminated, truncated, info = cls._step_model(action)
        reward = np.asarray(reward, dtype=np.float32)
        terminated, truncated = np.asarray(terminated), np.asarray(truncated)
        count = PhysicsManager._cfg.num_envs
        if reward.shape != (count,) or not np.isfinite(reward).all():
            raise ValueError("Backend reward must be a finite (num_envs,) array")
        if any(value.shape != (count,) or value.dtype != bool for value in (terminated, truncated)):
            raise ValueError("Backend done flags must be boolean (num_envs,) arrays")
        cls._store_observation(observation)
        cls._ready[:] = True
        cls.transition = {
            **copy.deepcopy(info),
            **{key: value.copy() for key, value in observation.items()},
            "reward": reward.copy(),
            "terminated": terminated.copy(),
            "truncated": truncated.copy(),
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
    num_envs: int = 1


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
        if type(cfg.scene) is not InteractiveSceneCfg or has_assets:
            raise ValueError("Neural backends require an empty InteractiveScene, without assets")
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
        return torch.as_tensor(transition["reward"], device=self.device).clone()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        transition = self.sim.physics_manager.transition
        terminated = torch.as_tensor(transition["terminated"], device=self.device).clone()
        truncated = torch.as_tensor(transition["truncated"], device=self.device).clone()
        return terminated, truncated | (self.episode_length_buf >= self.max_episode_length)

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)
        self.extras["reset_info"] = self.sim.physics_manager.reset_episode(
            seed=int(torch.initial_seed()),
            options=self.cfg.reset_options,
            env_ids=torch.as_tensor(env_ids).cpu().numpy(),
        )

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Read adapter pixels without invoking a model or a USD renderer."""
        if self.render_mode is None:
            return None
        image = self.sim.physics_manager.render()
        return None if image is None else image.copy()
