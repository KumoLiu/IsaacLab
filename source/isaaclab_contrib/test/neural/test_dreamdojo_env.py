# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight frontend contracts; real WM validation uses the GPU demo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from isaaclab_contrib.neural.neural_env import NeuralDirectEnv, NeuralPhysicsCfg, NeuralPhysicsManager

from isaaclab_tasks.contrib.dreamdojo_trocar import env_cfg as module
from isaaclab_tasks.contrib.dreamdojo_trocar.env_cfg import DreamDojoInferenceCfg, make_dreamdojo_env_cfg


@pytest.fixture
def cfg(tmp_path):
    checkpoint = tmp_path / "placeholder.pt"
    checkpoint.touch()
    return DreamDojoInferenceCfg(
        dreamdojo_root=str(tmp_path),
        dataset=str(tmp_path),
        wm_checkpoint=str(checkpoint),
        reward_checkpoint=str(checkpoint),
        lam_checkpoint=str(checkpoint),
        max_chunks=2,
    )


class FakeInference:
    """Deterministic fixture only; never available as a production backend."""

    def __init__(self, settings):
        self.cfg = SimpleNamespace(**settings)
        self.pid = os.getpid()
        self.count = np.zeros(settings["num_envs"], dtype=np.int64)
        self.closed = False
        self.requests = []

    def reset(self, env_ids, episodes, seeds):
        return self.request("reset", env_ids=env_ids.tolist(), episode_indices=episodes.tolist(), seeds=seeds.tolist())[
            0
        ]

    def step(self, action):
        return self.request("step", arrays={"action": action})[0]

    def info(self, env_ids=None):
        count = self.count if env_ids is None else self.count[env_ids]
        return {"is_success": np.zeros(len(count), dtype=bool), "milestone_stage": np.minimum(count, 2)}

    def request(self, command, *, arrays=None, **kwargs):
        action = (arrays or {}).get("action")
        self.requests.append((command, action, kwargs))
        indices = np.asarray(kwargs.get("env_ids", np.arange(self.cfg.num_envs)))
        if command == "reset":
            self.count[indices] = 0
        else:
            self.count += 1
        image = np.broadcast_to(self.count[indices, None, None, None], (len(indices), 480, 640, 3)).astype(np.uint8)
        arrays = {
            "main_images": image,
            "states": np.zeros((len(indices), 28), np.float32) if action is None else action[:, -1].copy(),
            "frames": np.repeat(image[:, None], 1 if command == "reset" else 6, axis=1),
        }
        if command == "step":
            arrays.update(
                reward=np.ones(self.cfg.num_envs, np.float32),
                command_rewards=np.tile(np.asarray([0.0] * 11 + [1.0], dtype=np.float32), (self.cfg.num_envs, 1)),
                probabilities=np.zeros((self.cfg.num_envs, 6, 3), dtype=np.float32),
                terminated=np.zeros(self.cfg.num_envs, dtype=bool),
                truncated=np.asarray(self.count >= self.cfg.max_chunks),
            )
        return arrays, self.info(indices)

    def close(self):
        self.closed = True


@pytest.fixture
def env(native_cfg):
    environment = NeuralDirectEnv(native_cfg, render_mode="rgb_array")
    yield environment
    environment.close()


@pytest.mark.parametrize("episode", [-1, 0.5, True])
def test_invalid_episode_never_reaches_model(cfg, episode):
    with pytest.raises(ValueError, match="episode_index"):
        make_dreamdojo_env_cfg(cfg, episode_index=episode)


@pytest.mark.parametrize("num_envs", [1, 3])
def test_rlinf_subproc_preserves_native_terminal_observation(cfg, num_envs):
    import torch

    SubProcIsaacLabEnv = pytest.importorskip("rlinf.envs.isaaclab.venv").SubProcIsaacLabEnv

    def factory():
        module.DreamDojoSimulator = FakeInference
        return NeuralDirectEnv(make_dreamdojo_env_cfg(cfg, num_envs=num_envs))

    process_env = SubProcIsaacLabEnv(factory, timeout_s=30)
    try:
        process_env.reset(seed=23, options={"episode_index": 3})
        action = torch.ones(num_envs, 12, 28)
        process_env.step(action)
        obs, reward, terminated, truncated, extras = process_env.step(action)
        assert obs["policy"]["main_images"].max() == 0
        assert extras["final_obs"]["policy"]["main_images"].max() == 2
        assert truncated[0] and not terminated[0] and reward[0] == 1
        assert extras["neural_transition"]["command_rewards"].sum() == num_envs
        assert obs["policy"]["states"].shape == (num_envs, 28)
    finally:
        process_env.close()
    assert not process_env.isaac_lab_process.is_alive()


@pytest.mark.parametrize("updates", [{"max_chunks": 0}, {"num_inference_steps": 0}])
def test_config_rejects_invalid_settings(cfg, updates):
    with pytest.raises(ValueError):
        replace(cfg, **updates).validate()


def test_missing_checkpoint_fails_before_launch(cfg):
    with pytest.raises(FileNotFoundError, match="wm_checkpoint"):
        replace(cfg, wm_checkpoint="/nonexistent/neural-demo-checkpoint.pt").validate()


@pytest.fixture
def native_cfg(cfg, monkeypatch):
    monkeypatch.setattr(module, "DreamDojoSimulator", FakeInference)
    return make_dreamdojo_env_cfg(cfg, episode_index=3)


def test_native_core_step_reset_final_observation_and_close(native_cfg, monkeypatch):
    import torch

    from isaaclab.envs import DirectRLEnv
    from isaaclab.sim import SimulationContext

    from isaaclab_tasks.contrib.dreamdojo_trocar.env_cfg import DreamDojoPhysicsManager

    # This test uses the REAL core environment, USD context and empty scene.
    # Only expensive neural inference is replaced with a deterministic fixture.
    def forbidden(*args, **kwargs):
        raise AssertionError("DreamDojo must not start another inference process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with gym.make("Isaac-DreamDojo-Trocar-Direct-v0", cfg=native_cfg, render_mode="rgb_array") as env:
        native = env.unwrapped
        assert isinstance(native, DirectRLEnv)
        assert isinstance(native.sim, SimulationContext)
        assert native.sim.physics_manager is DreamDojoPhysicsManager
        assert [key for key in native.scene.keys() if native.scene[key] is not None] == []
        assert native.sim.get_scene_data_provider().transform_count == 0
        with pytest.raises(RuntimeError, match="Reset"):
            native.step(torch.zeros(1, 12, 28))
        observation, _ = env.reset(seed=23)
        worker = native.sim.physics_manager._model
        assert worker.pid == os.getpid()
        assert worker.requests[-1][2] == {"env_ids": [0], "episode_indices": [3], "seeds": [23]}
        reset_image = observation["policy"]["main_images"].clone()
        action = torch.arange(336, dtype=torch.float32).reshape(1, 12, 28) / 100
        observation, reward, terminated, truncated, extras = env.step(action)
        np.testing.assert_array_equal(worker.requests[-1][1], action.numpy())
        assert extras["neural_transition"]["frames"].shape == (1, 6, 480, 640, 3)
        assert not terminated[0] and not truncated[0] and reward[0] == 1
        torch.testing.assert_close(observation["policy"]["states"], action[:, -1])
        assert int(native.episode_length_buf[0]) == 1
        assert native.sim.get_physics_step_count() == 1
        assert native.sim.physics_manager.get_simulation_time() == pytest.approx(0.4)
        assert worker.requests[-1][0] == "step"
        rendered = env.render()
        rendered[:] = 0
        assert env.render().any()
        native.sim.forward()
        assert worker.requests[-1][0] == "step" and len(worker.requests) == 2
        observation, _, terminated, truncated, extras = env.step(action)
        assert not terminated[0] and truncated[0]
        assert int(native.episode_length_buf[0]) == 0
        assert native.common_step_counter == native.sim.get_physics_step_count() == 2
        assert native.sim.physics_manager.get_simulation_time() == pytest.approx(0.8)
        torch.testing.assert_close(observation["policy"]["main_images"], reset_image)
        assert extras["final_obs"]["policy"]["main_images"].max() == 2
        assert extras["neural_transition"]["frames"].max() == 2
        assert worker.requests[-1][0] == "reset"
        # Same-Step autoreset allows step() again without an external reset.
        _, _, _, truncated, extras = env.step(action)
        assert not truncated[0] and int(native.episode_length_buf[0]) == 1
        assert "final_obs" not in extras
    native.close()  # Closing an already-closed native environment is safe.
    assert worker.closed
    assert SimulationContext.instance() is None
    assert DreamDojoPhysicsManager._model is None


@pytest.mark.parametrize("terminate_on_success", [False, True])
def test_native_full_success_is_not_positive_partial_reward(native_cfg, monkeypatch, terminate_on_success):
    import torch

    monkeypatch.setattr(
        FakeInference,
        "info",
        lambda self, env_ids=None: {"milestone_stage": np.array([3]), "is_success": np.array([True])},
    )
    native_cfg.sim.physics.terminate_on_success = terminate_on_success
    with gym.make("Isaac-DreamDojo-Trocar-Direct-v0", cfg=native_cfg) as env:
        env.reset()
        _, _, terminated, truncated, extras = env.step(torch.zeros(1, 12, 28))
        assert bool(terminated[0]) == terminate_on_success and not truncated[0]
        assert int(env.unwrapped.episode_length_buf[0]) == (0 if terminate_on_success else 1)
        assert extras["neural_transition"]["milestone_stage"] == 3


@pytest.mark.parametrize("setting", ["num_envs", "decimation", "episode_length_s", "robot"])
def test_native_rejects_unsupported_layout_before_startup(native_cfg, setting):
    from isaaclab.sim import SimulationContext

    if setting == "num_envs":
        native_cfg.scene.num_envs = 2
    elif setting == "robot":
        native_cfg.scene.robot = object()
    else:
        setattr(native_cfg, setting, 3)
    with pytest.raises(ValueError):
        NeuralDirectEnv(native_cfg)
    assert SimulationContext.instance() is None


def test_native_model_startup_failure_cleans_core_singleton(native_cfg, monkeypatch):
    from isaaclab.sim import SimulationContext

    def fail(settings):
        raise RuntimeError("model startup failed")

    monkeypatch.setattr(module, "DreamDojoSimulator", fail)
    with pytest.raises(RuntimeError, match="model startup failed"):
        NeuralDirectEnv(native_cfg)
    assert SimulationContext.instance() is None


@pytest.mark.parametrize("fail_load", [False, True])
def test_inprocess_load_restores_cwd_and_lam_environment(cfg, monkeypatch, fail_load):
    from dataclasses import asdict

    from isaaclab_tasks.contrib.dreamdojo_trocar.simulator import DreamDojoSimulator

    original_cwd = Path.cwd()
    monkeypatch.setenv("DREAMDOJO_LAM_CHECKPOINT", "parent-setting")

    def load_pipe(**kwargs):
        assert Path.cwd() == Path(cfg.dreamdojo_root)
        assert os.environ["DREAMDOJO_LAM_CHECKPOINT"] == cfg.lam_checkpoint
        assert kwargs["config_file"] == "isaaclab_tasks/contrib/dreamdojo_trocar/env_cfg.py"
        if fail_load:
            raise RuntimeError("checkpoint load failed")
        return object()

    for name, symbols in {
        "cosmos_predict2._src.predict2.inference.video2world": {"Video2WorldInference": load_pipe},
        "isaaclab_tasks.contrib.dreamdojo_trocar.actions": {"G1DreamDojoActionBridge": lambda *a: object()},
        "isaaclab_tasks.contrib.dreamdojo_trocar.dataset": {"LeRobotV21InitDataset": lambda *a, **kw: object()},
        "isaaclab_tasks.contrib.dreamdojo_trocar.reward": {"BatchedMilestoneReward": lambda *a, **kw: object()},
    }.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**symbols))
    if fail_load:
        with pytest.raises(RuntimeError, match="checkpoint load failed"):
            DreamDojoSimulator(asdict(cfg))
    else:
        model = DreamDojoSimulator(asdict(cfg))
        model.close()
        model.close()
        assert not model.ready and not hasattr(model, "pipe")
        with pytest.raises(RuntimeError, match="reset"):
            model.step(np.zeros((1, 12, 28), dtype=np.float32))
    assert Path.cwd() == original_cwd
    assert os.environ["DREAMDOJO_LAM_CHECKPOINT"] == "parent-setting"


class ToyPhysicsManager(NeuralPhysicsManager):
    """A different in-process model, without a Gym wrapper, only for tests."""

    @classmethod
    def _load_model(cls):
        from isaaclab.physics import PhysicsManager

        cls.closed = False
        cls.calls = np.zeros(PhysicsManager._cfg.num_envs, dtype=np.int64)

    @classmethod
    def _reset_model(cls, *, env_ids, seed, options):
        cls.calls[env_ids] = 0
        cls.end_flag = options.get("end_flag")
        return {"latent": np.tile(np.array([options["initial_value"], 0], dtype=np.float32), (len(env_ids), 1))}, {
            "seed": seed
        }

    @classmethod
    def _step_model(cls, action):
        cls.calls += 1
        value = action.mean(axis=tuple(range(1, action.ndim)))
        return (
            {"latent": np.stack([value, cls.calls], axis=1).astype(np.float32)},
            value,
            np.full(len(cls.calls), cls.end_flag == "terminated") | (value < 0),
            np.full(len(cls.calls), cls.end_flag == "truncated"),
            {"model_calls": cls.calls.copy()},
        )

    @classmethod
    def render(cls):
        return np.full((4, 5, 3), cls.calls[0], dtype=np.uint8)

    @classmethod
    def _close_model(cls):
        cls.closed = True


def make_generic_cfg(action_shape=(2, 3), step_dt=0.2, end_flag=None, num_envs=1):
    from isaaclab_contrib.neural.neural_env import NeuralDirectEnvCfg

    physics = NeuralPhysicsCfg(
        class_type=ToyPhysicsManager,
        action_space=gym.spaces.Box(-2, 2, action_shape, dtype=np.float32),
        observation_space=gym.spaces.Dict({"latent": gym.spaces.Box(-10, 10, (2,), dtype=np.float32)}),
        step_dt=step_dt,
        num_envs=num_envs,
    )
    cfg = NeuralDirectEnvCfg(
        action_space=physics.action_space,
        observation_space=physics.observation_space,
        episode_length_s=2 * step_dt,
        reset_options={"initial_value": 0.25, "end_flag": end_flag},
    )
    cfg.sim.dt = step_dt
    cfg.sim.physics = physics
    cfg.scene.num_envs = num_envs
    return cfg


def test_multi_env_partial_autoreset_keeps_other_rows_and_shared_time():
    import torch

    with NeuralDirectEnv(make_generic_cfg(num_envs=3)) as env:
        env.reset(seed=29)
        action = torch.tensor([-0.5, 0.5, 0.75])[:, None, None].expand(3, 2, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward.tolist() == [-0.5, 0.5, 0.75]
        assert terminated.tolist() == [True, False, False] and not truncated.any()
        assert obs["policy"]["latent"].tolist() == [[0.25, 0], [0.5, 1], [0.75, 1]]
        assert info["final_obs"]["policy"]["latent"][0].tolist() == [-0.5, 1]
        assert env.episode_length_buf.tolist() == [0, 1, 1]
        obs, _, _, truncated, info = env.step(action.abs())
        assert truncated.tolist() == [False, True, True]
        assert env.episode_length_buf.tolist() == [1, 0, 0]
        assert obs["policy"]["latent"].tolist() == [[0.5, 1], [0.25, 0], [0.25, 0]]
        assert info["neural_transition"]["model_calls"].tolist() == [1, 2, 2]
        assert env.sim.physics_manager.get_simulation_time() == pytest.approx(0.4)
        assert env.common_step_counter == env.sim.get_physics_step_count() == 2


@pytest.mark.parametrize("use_lora", [False, True])
def test_dreamdojo_sampler_batch_matches_individual_rows(use_lora):
    """Exercise the real CPU scheduler, without loading any WM weights."""
    import torch

    module = pytest.importorskip("cosmos_predict2._src.predict2.models.text2world_model_rectified_flow")
    model_type = module.Text2WorldModelRectifiedFlow
    method = model_type.generate_samples_from_batch_lora if use_lora else model_type.generate_samples_from_batch

    def sample(seeds, biases):
        model = SimpleNamespace(
            _normalize_video_databatch_inplace=lambda batch: None,
            _augment_image_dim_inplace=lambda batch: None,
            is_image_batch=lambda batch: False,
            input_data_key="video",
            tensor_kwargs={"device": "cpu"},
            config=SimpleNamespace(use_kerras_sigma_at_inference=False, use_lora=use_lora),
            net=SimpleNamespace(is_context_parallel_enabled=False),
            sample_scheduler=module.FlowUniPCMultistepScheduler(),
            _inference_noise=model_type._inference_noise,
            get_velocity_fn_from_batch=lambda batch, *args, **kwargs: (
                lambda noise, latents, t: latents * 0.2 + noise * 0.1 + batch["bias"]
            ),
        )
        return method(
            model,
            {"bias": torch.tensor(biases)[:, None, None, None, None]},
            state_shape=(2, 2, 2, 2),
            n_sample=len(seeds),
            sample_seeds=seeds,
            num_steps=5,
        )

    batch = sample([17, 29], [0.1, 0.7])
    singles = torch.cat([sample([17], [0.1]), sample([29], [0.7])])
    torch.testing.assert_close(batch, singles, rtol=0, atol=0)
    torch.testing.assert_close(sample([29, 17], [0.7, 0.1]), batch.flip(0), rtol=0, atol=0)
    shared = sample([17, 17], [0.1, 0.1])
    torch.testing.assert_close(shared[0], shared[1], rtol=0, atol=0)


@pytest.mark.parametrize("action_shape,step_dt", [((2, 3), 0.2), ((4, 1), 0.5)])
def test_generic_model_spaces_timing_policy_swap_and_native_reset(action_shape, step_dt):
    import torch

    from isaaclab.sim import SimulationContext

    from isaaclab_contrib.neural.neural_env import NeuralPolicy

    cfg = make_generic_cfg(action_shape, step_dt)
    with NeuralDirectEnv(cfg, render_mode="rgb_array") as env:
        backend = env.sim.physics_manager
        observation, info = env.reset(seed=29)
        assert info["reset_info"]["seed"] == 29
        assert observation["policy"]["latent"].shape == (1, 2)

        def feedback_policy(obs):
            return obs["latent"][:, :1].reshape(1, 1, 1).expand(1, *action_shape)

        def constant_policy(obs):
            return torch.full((1, *action_shape), 0.75)

        policy: NeuralPolicy = feedback_policy
        observation, reward, terminated, truncated, _ = env.step(policy(observation["policy"]))
        assert reward.item() == 0.25 and not terminated[0] and not truncated[0]
        assert backend.calls == 1
        rendered = env.render()
        rendered[:] = 0
        env.sim.forward()
        assert env.render().max() == 1 and backend.calls == 1
        policy = constant_policy  # No change to the model or native environment.
        observation, reward, terminated, truncated, extras = env.step(policy(observation["policy"]))
        assert reward.item() == 0.75 and not terminated[0] and truncated[0]
        assert observation["policy"]["latent"][0, 0] == 0.25
        assert extras["final_obs"]["policy"]["latent"][0, 0] == 0.75
        assert extras["neural_transition"]["model_calls"] == 2
        assert env.sim.physics_manager.get_simulation_time() == pytest.approx(2 * step_dt)
        assert int(env.episode_length_buf[0]) == 0
    assert backend.closed and SimulationContext.instance() is None


@pytest.mark.parametrize("end_flag", ["terminated", "truncated"])
def test_generic_backend_done_flags_are_preserved(end_flag):
    import torch

    with NeuralDirectEnv(make_generic_cfg(end_flag=end_flag)) as env:
        env.reset()
        _, _, terminated, truncated, extras = env.step(torch.zeros(1, 2, 3))
        assert bool(terminated[0]) == (end_flag == "terminated")
        assert bool(truncated[0]) == (end_flag == "truncated")
        assert "final_obs" in extras and int(env.episode_length_buf[0]) == 0


def test_generic_invalid_actions_do_not_advance_model():
    import torch

    with NeuralDirectEnv(make_generic_cfg()) as env:
        env.reset()
        for action in (
            torch.zeros(28),
            torch.zeros(2, 3),
            torch.zeros(1, 12, 28),
            torch.full((1, 2, 3), float("nan")),
            torch.full((1, 2, 3), float("inf")),
            torch.full((1, 2, 3), 3.0),
        ):
            with pytest.raises(ValueError):
                env.step(action)
            assert env.sim.physics_manager.calls == 0


@pytest.mark.parametrize("mismatch", ["time", "space"])
def test_generic_bad_contract_fails_before_model_load(mismatch, monkeypatch):
    from isaaclab.sim import SimulationContext

    cfg = make_generic_cfg()
    loaded = []
    monkeypatch.setattr(ToyPhysicsManager, "_load_model", classmethod(lambda cls: loaded.append(True)))
    if mismatch == "time":
        cfg.sim.dt = 0.1
    else:
        cfg.action_space = gym.spaces.Box(-2, 2, (3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="match"):
        NeuralDirectEnv(cfg)
    assert not loaded and SimulationContext.instance() is None


@pytest.mark.parametrize("bad_field", ["observation", "reward"])
def test_failed_transition_requires_reset(env, monkeypatch, bad_field):
    import torch

    original = FakeInference.request

    def bad_step(self, command, **kwargs):
        arrays, info = original(self, command, **kwargs)
        if command == "step":
            arrays["states" if bad_field == "observation" else "reward"] = np.asarray(np.nan, dtype=np.float32)
        return arrays, info

    env.reset()
    monkeypatch.setattr(FakeInference, "request", bad_step)
    with pytest.raises(ValueError):
        env.step(torch.zeros(1, 12, 28))
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(torch.zeros(1, 12, 28))
    monkeypatch.setattr(FakeInference, "request", original)
    env.reset()
    env.step(torch.zeros(1, 12, 28))


def test_neural_backend_and_task_runtime_import_without_rlinf():
    demo = Path(__file__).resolve().parents[4] / "scripts/demos/neural_sim.py"
    code = """
import contextlib
import importlib.abc
import io
import runpy
import sys
class NoRLinf(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rlinf', 'cosmos_predict2'):
            raise RuntimeError('Task registration/config import must not load RLinf or DreamDojo')
sys.meta_path.insert(0, NoRLinf())
from isaaclab_contrib.neural.neural_env import NeuralDirectEnv
from isaaclab_tasks.contrib.dreamdojo_trocar.env_cfg import DreamDojoPhysicsManager
from isaaclab_tasks.contrib.dreamdojo_trocar.simulator import DreamDojoSimulator
from isaaclab_tasks.contrib.dreamdojo_trocar.dataset import LeRobotV21InitDataset
from isaaclab_tasks.contrib.dreamdojo_trocar.actions import G1DreamDojoActionBridge
from isaaclab_tasks.contrib.dreamdojo_trocar.reward import BatchedMilestoneReward
demo = runpy.run_path(sys.argv[1])
sys.argv = [sys.argv[1], '--help']
help_output = io.StringIO()
with contextlib.redirect_stdout(help_output):
    try:
        demo['main']()
    except SystemExit as exc:
        assert exc.code == 0
assert '--dataset' in help_output.getvalue()
assert not any(n == 'rlinf' or n.startswith('rlinf.') for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(demo)], check=True, capture_output=True, text=True, timeout=60)


def test_same_neural_environment_with_existing_rsl_rl_wrapper():
    """Check a second framework's boundary, not policy training or WM quality."""
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    with NeuralDirectEnv(make_generic_cfg(action_shape=(2,))) as native:
        env = RslRlVecEnvWrapper(native)
        obs, _ = env.reset()
        assert obs["policy"]["latent"].shape == (1, 2)
        obs, reward, dones, _ = env.step(torch.full((1, 2), 0.5))
        assert reward.item() == 0.5 and not dones.any()
        _, _, dones, extras = env.step(torch.full((1, 2), 0.75))
        assert dones.all()
        assert extras["final_obs"]["policy"]["latent"][0, 0].item() == 0.75


def test_task_action_conditioning_preserves_layout_time_and_state_prefix(tmp_path):
    import torch

    from isaaclab_tasks.contrib.dreamdojo_trocar.actions import G1DreamDojoActionBridge

    stats = {key: {"min": [-1.0] * 43, "max": [1.0] * 43} for key in ("action", "observation.state")}
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    bridge = G1DreamDojoActionBridge(path, "cpu")
    actions = torch.arange(12).float()[None, :, None] + torch.arange(28).float()[None, None, :]
    state = torch.arange(28).float()[None] + 30
    encoded = bridge.encode_policy_chunk(torch.full((1, 28), -2.0), actions, state)
    expected = torch.zeros(1, 12, 384)
    # Independent reference: raw frames -2,0,2,...,22; hold command 11 past the executed horizon.
    samples = [torch.full((1, 28), -2.0)] + [actions[:, min(frame, 11)] for frame in range(0, 24, 2)]
    for src, dst in ((0, 15), (7, 29), (14, 22), (21, 36)):
        for frame in range(12):
            expected[:, frame, 58 + dst : 58 + dst + 7] = (
                samples[frame + 1][:, src : src + 7] - samples[(frame // 4) * 4][:, src : src + 7]
            )
    expected[:, 0, 15:22] = state[:, :7]
    expected[:, 0, 22:29] = state[:, 14:21]
    torch.testing.assert_close(encoded, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    [
        "plain",
        "task_fallback",
        "task_override",
        "missing_state",
        "bad_shape",
        "nonfinite",
        "short_episode",
        "wrong_fps",
    ],
)
def test_task_reset_dataset_uses_frame_two_and_causal_frame_zero_history(tmp_path, monkeypatch, case):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from isaaclab_tasks.contrib.dreamdojo_trocar.dataset import LeRobotV21InitDataset

    (tmp_path / "meta").mkdir()
    info = {
        "fps": 15 if case == "wrong_fps" else 30,
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": "episode_{episode_index:06d}.parquet",
        "video_path": "{video_key}.mp4",
    }
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    (tmp_path / "meta/modality.json").write_text(json.dumps({"video": {"head_view": {"original_key": "head"}}}))
    state = np.arange(112, dtype=np.float32).reshape(4, 28)
    actions = state + 200
    if case == "bad_shape":
        state = state[:, :27]
    elif case == "nonfinite":
        state[0, 0] = np.nan
    columns = {"observation.state": state.tolist(), "action": actions.tolist()}
    if case == "missing_state":
        del columns["observation.state"]
    elif case in ("task_fallback", "task_override"):
        columns["task_index"] = [7] * len(state)
        (tmp_path / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 7, "task": "fallback"}) + "\n")
        if case == "task_override":
            (tmp_path / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["episode"]}) + "\n")
    table = pa.table(columns)
    pq.write_table(table.slice(0, 2) if case == "short_episode" else table, tmp_path / "episode_000000.parquet")
    decoded = []

    def frames(self, path, indices):
        decoded.extend(indices)
        return np.full((len(indices), 2, 3, 3), 127, dtype=np.uint8)

    monkeypatch.setattr(LeRobotV21InitDataset, "_decode_video_frames", frames)
    expected_error = {
        "missing_state": "missing required column",
        "bad_shape": "finite 28-D",
        "nonfinite": "finite 28-D",
        "short_episode": "at least 3 frames",
        "wrong_fps": "30-Hz recording",
    }.get(case)
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            LeRobotV21InitDataset(tmp_path)[0]
        assert not decoded
        return
    data = LeRobotV21InitDataset(tmp_path, image_size=None)
    item = data[0]
    assert decoded == [2] and item["start_frame"] == 2
    assert item["episode_index"] == 0
    assert item["task"] == {"task_fallback": "fallback", "task_override": "episode"}.get(case, "")
    torch.testing.assert_close(item["image"], torch.full((3, 2, 3), 127 / 255))
    torch.testing.assert_close(item["state"], torch.from_numpy(state[2]))
    torch.testing.assert_close(item["previous_state"], torch.from_numpy(state[0]))
    torch.testing.assert_close(item["previous_action"], torch.from_numpy(actions[0]))


def test_task_reward_keeps_ordered_once_only_transitions_and_independent_resets():
    import torch

    from isaaclab_tasks.contrib.dreamdojo_trocar.reward import BatchedMilestoneReward

    reward = BatchedMilestoneReward.__new__(BatchedMilestoneReward)
    reward.device, reward.num_envs = torch.device("cpu"), 2
    reward.reset()
    for _ in range(16):
        assert not reward.advance(torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])).any()
    payouts = [reward.advance(torch.ones(2, 3)) for _ in range(13)]
    torch.testing.assert_close(torch.stack(payouts).sum(0), torch.tensor([3.0, 3.0]))
    assert not reward.advance(torch.ones(2, 3)).any()
    reward.reset(torch.tensor([True, False]))
    assert reward.stage.tolist() == [0, 3]
