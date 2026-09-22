# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DreamDojo backend: in-process model inference and native lifecycle hooks."""

from __future__ import annotations

import math
import time
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

from isaaclab.physics import PhysicsManager
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_contrib.neural.neural_env import NeuralDirectEnvCfg, NeuralPhysicsCfg, NeuralPhysicsManager

from .simulator import DreamDojoSimulator

if TYPE_CHECKING:
    from cosmos_predict2._src.predict2.action.configs.action_conditioned.config import Config as DreamDojoConfig


def make_config() -> DreamDojoConfig:
    """DreamDojo loader entrypoint: register the task recipe over official G1 defaults."""
    from cosmos_predict2._src.predict2.action.configs.action_conditioned.config import (
        make_config as make_dreamdojo_config,
    )
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    config = make_dreamdojo_config()
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name="dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora",
        node=OmegaConf.load(Path(__file__).parent / "config/dreamdojo_wm.yaml"),
    )
    return config


@dataclass
class DreamDojoInferenceCfg:
    """Explicit paths and inference settings for the robot-free demo.

    Each step executes 12 absolute joint targets [rad] at 30 Hz (0.4 s).
    ``max_chunks`` is a time limit, not a diffusion or policy iteration count.
    """

    dreamdojo_root: str
    dataset: str
    wm_checkpoint: str
    reward_checkpoint: str
    lam_checkpoint: str
    experiment: str = "dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora"
    max_chunks: int = 10
    num_inference_steps: int = 15
    seed: int = 1234

    def validate(self) -> None:
        """Fail before starting CUDA when an input is missing or invalid."""
        for name in ("wm_checkpoint", "reward_checkpoint", "lam_checkpoint"):
            if not Path(getattr(self, name)).is_file():
                raise FileNotFoundError(f"{name}: {getattr(self, name)}")
        for name in ("dreamdojo_root", "dataset"):
            if not Path(getattr(self, name)).is_dir():
                raise FileNotFoundError(f"{name}: {getattr(self, name)}")
        if self.max_chunks < 1 or self.num_inference_steps < 1:
            raise ValueError("max_chunks and num_inference_steps must be positive")


class DreamDojoPhysicsManager(NeuralPhysicsManager):
    """Own DreamDojo inference in the native environment's process."""

    _model: DreamDojoSimulator | None = None

    @classmethod
    def validate_env_cfg(cls, cfg: NeuralDirectEnvCfg) -> None:
        super().validate_env_cfg(cfg)
        if not isinstance(cfg.sim.physics, DreamDojoPhysicsCfg) or not math.isclose(cfg.sim.dt, 0.4):
            raise ValueError("DreamDojo requires DreamDojoPhysicsCfg and sim.dt=0.4 s")
        if not math.isclose(cfg.episode_length_s, cfg.sim.physics.inference.max_chunks * 0.4):
            raise ValueError("episode_length_s must equal inference.max_chunks * 0.4")
        cls._reset_values(cfg.reset_options, cfg.scene.num_envs, cfg.seed)

    @staticmethod
    def _reset_values(options: dict, num_envs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        episodes = options.get("episode_indices", [options.get("episode_index", 0)] * num_envs)
        seeds = options.get("seeds", [(seed + i) % 2**32 for i in range(num_envs)])
        for name, values in (("episode_index / episode_indices", episodes), ("seeds", seeds)):
            if len(values) != num_envs or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"{name} must contain {num_envs} nonnegative integers")
        if any(s >= 2**32 for s in seeds):
            raise ValueError("seeds must be smaller than 2**32")
        return np.asarray(episodes, dtype=np.int64), np.asarray(seeds, dtype=np.int64)

    @classmethod
    def _load_model(cls) -> None:
        cls._model = None
        cfg = PhysicsManager._cfg.inference
        cfg.validate()
        cls._model = DreamDojoSimulator({**asdict(cfg), "num_envs": PhysicsManager._cfg.num_envs})

    @staticmethod
    def _unpack(arrays: dict, info: dict) -> tuple[dict, dict]:
        observation = {key: arrays.pop(key) for key in ("main_images", "states")}
        info.update(arrays)
        info["state_source"] = "recorded_reset_then_last_command_proxy"
        return observation, info

    @classmethod
    def _reset_model(cls, *, env_ids: np.ndarray, seed: int, options: dict) -> tuple[dict, dict]:
        start = time.perf_counter()
        episodes, seeds = cls._reset_values(options, PhysicsManager._cfg.num_envs, seed)
        arrays = cls._model.reset(env_ids, episodes[env_ids], seeds[env_ids])
        info = {**cls._model.info(env_ids), "env_ids": env_ids.copy(), "wall_seconds": time.perf_counter() - start}
        return cls._unpack(arrays, info)

    @classmethod
    def _step_model(cls, action: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        start = time.perf_counter()
        arrays = cls._model.step(action)
        info = {**cls._model.info(), "wall_seconds": time.perf_counter() - start}
        reward = arrays.pop("reward")
        terminated = arrays.pop("terminated")
        truncated = arrays.pop("truncated")
        if PhysicsManager._cfg.terminate_on_success:
            terminated |= info["milestone_stage"] == 3
        observation, info = cls._unpack(arrays, info)
        return observation, reward, terminated, truncated, info

    @classmethod
    def render(cls) -> np.ndarray | None:
        return None if cls.observation is None else cls.observation["main_images"][0].cpu().numpy().copy()

    @classmethod
    def _close_model(cls) -> None:
        if cls._model is not None:
            cls._model.close()
        cls._model = None


@configclass
class DreamDojoPhysicsCfg(NeuralPhysicsCfg):
    """DreamDojo input/output contract and optional three-stage success termination."""

    class_type: type[PhysicsManager] = DreamDojoPhysicsManager
    action_space: spaces.Box = spaces.Box(-np.inf, np.inf, (12, 28), dtype=np.float32)
    observation_space: spaces.Dict = spaces.Dict(
        {
            "main_images": spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
            "states": spaces.Box(-np.inf, np.inf, (28,), dtype=np.float32),
        }
    )
    step_dt: float = 0.4
    inference: DreamDojoInferenceCfg = MISSING
    terminate_on_success: bool = False


def make_dreamdojo_env_cfg(
    inference: DreamDojoInferenceCfg, *, episode_index: int = 0, num_envs: int = 1, terminate_on_success: bool = False
) -> NeuralDirectEnvCfg:
    """Build a plain native config for one 0.4 s transition per action chunk."""
    physics = DreamDojoPhysicsCfg(inference=inference, num_envs=num_envs, terminate_on_success=terminate_on_success)
    cfg = NeuralDirectEnvCfg(
        sim=SimulationCfg(
            device="cpu", dt=physics.step_dt, physics=physics, create_stage_in_memory=True, visualizer_cfgs=[]
        ),
        action_space=physics.action_space,
        observation_space=physics.observation_space,
        episode_length_s=inference.max_chunks * physics.step_dt,
        seed=inference.seed,
        reset_options={"episode_index": episode_index},
    )
    cfg.scene.num_envs = num_envs
    DreamDojoPhysicsManager.validate_env_cfg(cfg)
    return cfg
