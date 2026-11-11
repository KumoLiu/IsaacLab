# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos3 model configuration and native Isaac Lab lifecycle hooks."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from gymnasium import spaces

from isaaclab.physics import PhysicsManager
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_contrib.neural.neural_env import NeuralDirectEnvCfg, NeuralPhysicsCfg, NeuralPhysicsManager

from .simulator import Cosmos3Simulator

Cosmos3Reward = Callable[[np.ndarray, dict], tuple[np.ndarray, np.ndarray, dict]]
"""Task callback: RGB chunks (N,T,H,W,3) -> rewards (N,), terminal flags (N,), metrics."""


@dataclass
class Cosmos3InferenceCfg:
    """Load one local unified-v1 checkpoint; actions are already normalized 59D controls.

    ``window`` conditions each prediction on the last generated image (Stage 0).
    ``session`` retains the native causal/decoder history (distilled checkpoints).
    ``chunk_size / fps`` is one environment step [s]; image_size is decoded (H,W).
    Window defaults match the native forward-dynamics API: 30 denoising steps
    and shift 10 when the corresponding overrides are None.
    """

    checkpoint_path: str
    output_dir: str
    config_file: str | None = None
    mode: Literal["window", "session"] = "window"
    image_size: tuple[int, int] = (480, 640)
    chunk_size: int = 16
    fps: int = 30
    domain_name: str = "hand_pose"
    resolution: str = "480"
    num_inference_steps: int | None = None
    guidance: float = 1.0
    shift: float | None = None
    use_ema_weights: bool = True

    def validate(self) -> None:
        """Check paths and declared spaces before allocating model resources."""
        if not Path(self.checkpoint_path).is_dir():
            raise FileNotFoundError(f"checkpoint_path: {self.checkpoint_path}")
        if self.config_file is not None and not Path(self.config_file).is_file():
            raise FileNotFoundError(f"config_file: {self.config_file}")
        if self.mode not in ("window", "session"):
            raise ValueError("mode must be 'window' or 'session'")
        for name in ("chunk_size", "fps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_inference_steps is not None and (
            type(self.num_inference_steps) is not int or self.num_inference_steps < 1
        ):
            raise ValueError("num_inference_steps must be a positive integer or None")
        if self.mode == "session" and (
            self.num_inference_steps is not None or self.guidance != 1 or self.shift is not None
        ):
            raise ValueError(
                "session mode uses the checkpoint sampler; leave num_inference_steps/shift=None and guidance=1"
            )
        if self.chunk_size % 4:
            raise ValueError("chunk_size must be divisible by the temporal compression factor 4")
        if len(self.image_size) != 2 or any(type(v) is not int or v < 1 for v in self.image_size):
            raise ValueError("image_size must contain positive decoded height and width")
        if not self.domain_name.strip() or not self.resolution.strip() or not self.output_dir.strip():
            raise ValueError("domain_name, resolution and output_dir must be nonempty")
        if self.shift is not None and (not math.isfinite(self.shift) or self.shift <= 0):
            raise ValueError("shift must be positive and finite, or None")
        if not math.isfinite(self.guidance) or self.guidance < 0:
            raise ValueError("guidance must be finite and nonnegative")


def _reset_values(options: dict, num_envs: int, seed: int) -> tuple[list[Path], list[str], np.ndarray]:
    images = options.get("initial_images", [options.get("initial_image")] * num_envs)
    prompts = options.get("prompts", [options.get("prompt")] * num_envs)
    seeds = options.get("seeds", [(seed + i) % 2**32 for i in range(num_envs)])
    if not isinstance(images, (list, tuple)) or len(images) != num_envs:
        raise ValueError("initial_images must contain one image path per environment")
    if not isinstance(prompts, (list, tuple)) or len(prompts) != num_envs:
        raise ValueError("prompts must contain one task description per environment")
    if any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Each prompt must be a nonempty string")
    if (
        not isinstance(seeds, (list, tuple))
        or len(seeds) != num_envs
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("seeds must contain one integer in [0,2**32) per environment")
    paths = []
    for image in images:
        if not isinstance(image, (str, Path)) or not Path(image).is_file():
            raise FileNotFoundError(f"initial_image: {image}")
        paths.append(Path(image))
    return paths, list(prompts), np.asarray(seeds, dtype=np.int64)


class Cosmos3PhysicsManager(NeuralPhysicsManager):
    """Use one in-process Cosmos model; Isaac Lab owns episode length and autoreset."""

    _model: Cosmos3Simulator | None = None

    @classmethod
    def validate_env_cfg(cls, cfg: NeuralDirectEnvCfg) -> None:
        super().validate_env_cfg(cfg)
        physics = cfg.sim.physics
        if not isinstance(physics, Cosmos3PhysicsCfg):
            raise ValueError("Cosmos3 requires Cosmos3PhysicsCfg")
        physics.inference.validate()
        if physics.inference.mode == "session" and physics.num_envs != 1:
            raise ValueError("Native Cosmos3 sessions support one environment per process")
        if not math.isclose(cfg.sim.dt, physics.inference.chunk_size / physics.inference.fps):
            raise ValueError("sim.dt must equal chunk_size / fps [s]")
        inference = physics.inference
        action_space = spaces.Box(-np.inf, np.inf, (inference.chunk_size, 59), np.float32)
        observation_space = spaces.Dict({"main_images": spaces.Box(0, 255, (*inference.image_size, 3), np.uint8)})
        if physics.action_space != action_space or physics.observation_space != observation_space:
            raise ValueError("Cosmos3 spaces must match the declared chunk_size, 59D controls and RGB canvas")
        if physics.reward_fn is not None and not callable(physics.reward_fn):
            raise ValueError("reward_fn must be callable or None")
        _reset_values(cfg.reset_options, physics.num_envs, cfg.seed)

    @classmethod
    def _load_model(cls) -> None:
        cls._model = Cosmos3Simulator(PhysicsManager._cfg.inference, PhysicsManager._cfg.num_envs)
        cls._model.load()

    @classmethod
    def _reset_model(cls, *, env_ids: np.ndarray, seed: int, options: dict) -> tuple[dict, dict]:
        images, prompts, seeds = _reset_values(options, PhysicsManager._cfg.num_envs, seed)
        frames = cls._model.reset(env_ids, images, prompts, seeds)
        return {"main_images": frames[:, -1]}, cls._model.info(env_ids)

    @classmethod
    def _step_model(cls, action: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        frames = cls._model.step(action)
        info = {**cls._model.info(), "frames": frames}
        count = PhysicsManager._cfg.num_envs
        reward, terminated = np.zeros(count, np.float32), np.zeros(count, bool)
        callback = PhysicsManager._cfg.reward_fn
        if callback is not None:
            reward, terminated, metrics = callback(frames.copy(), cls._model.info())
            info["task_metrics"] = metrics
        return {"main_images": frames[:, -1]}, reward, terminated, np.zeros(count, bool), info

    @classmethod
    def render(cls) -> np.ndarray | None:
        return None if cls.observation is None else cls.observation["main_images"][0].cpu().numpy().copy()

    @classmethod
    def _close_model(cls) -> None:
        try:
            if cls._model is not None:
                cls._model.close()
        finally:
            cls._model = None


@configclass
class Cosmos3PhysicsCfg(NeuralPhysicsCfg):
    """Cosmos model I/O and optional task reward; no classifier or robot is hard-coded."""

    class_type: type[PhysicsManager] = Cosmos3PhysicsManager
    inference: Cosmos3InferenceCfg = MISSING
    reward_fn: Cosmos3Reward | None = None


def make_cosmos3_env_cfg(
    inference: Cosmos3InferenceCfg,
    *,
    initial_image: str,
    prompt: str,
    num_envs: int = 1,
    max_chunks: int = 60,
    seed: int = 1234,
    reward_fn: Cosmos3Reward | None = None,
) -> NeuralDirectEnvCfg:
    """Build image-only observations and (chunk_size, 59) normalized action spaces.

    Use reset options ``initial_images``, ``prompts`` and ``seeds`` to select
    distinct initial conditions per environment. A task callback supplies rewards
    and success/failure termination; without it rewards are zero.
    """
    inference.validate()
    if type(max_chunks) is not int or max_chunks < 1:
        raise ValueError("max_chunks must be a positive integer")
    physics = Cosmos3PhysicsCfg(
        inference=inference,
        num_envs=num_envs,
        action_space=spaces.Box(-np.inf, np.inf, (inference.chunk_size, 59), np.float32),
        observation_space=spaces.Dict({"main_images": spaces.Box(0, 255, (*inference.image_size, 3), np.uint8)}),
        step_dt=inference.chunk_size / inference.fps,
        reward_fn=reward_fn,
    )
    cfg = NeuralDirectEnvCfg(
        sim=SimulationCfg(
            device="cpu", dt=physics.step_dt, physics=physics, create_stage_in_memory=True, visualizer_cfgs=[]
        ),
        action_space=physics.action_space,
        observation_space=physics.observation_space,
        # Prevent ceil from adding a chunk after the seconds-to-steps round trip.
        episode_length_s=math.nextafter(max_chunks * physics.step_dt, 0.0),
        seed=seed,
        reset_options={"initial_image": initial_image, "prompt": prompt},
    )
    cfg.scene.num_envs = num_envs
    Cosmos3PhysicsManager.validate_env_cfg(cfg)
    return cfg
