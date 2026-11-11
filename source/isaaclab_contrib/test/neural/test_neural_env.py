# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Model-independent lifecycle, spaces and independent resets through native Isaac Lab."""

import gymnasium as gym
import numpy as np
import pytest

from isaaclab_contrib.neural.neural_env import NeuralDirectEnv, NeuralPhysicsCfg, NeuralPhysicsManager


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


@pytest.mark.parametrize("action_shape,step_dt", [((2, 3), 0.2), ((4, 1), 0.5)])
def test_generic_model_spaces_timing_policy_swap_and_native_reset(action_shape, step_dt):
    import torch

    from isaaclab.sim import SimulationContext

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

        policy = feedback_policy
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


@pytest.mark.parametrize("setting", ["num_envs", "decimation", "device", "robot"])
def test_unsupported_scene_layout_is_rejected_before_startup(setting):
    from isaaclab.sim import SimulationContext

    cfg = make_generic_cfg()
    if setting == "num_envs":
        cfg.scene.num_envs = 2
    elif setting == "decimation":
        cfg.decimation = 2
    elif setting == "device":
        cfg.sim.device = "cuda:0"
    else:
        cfg.scene.robot = object()
    with pytest.raises(ValueError):
        NeuralDirectEnv(cfg)
    assert SimulationContext.instance() is None


@pytest.mark.parametrize("bad_field", ["observation", "reward", "terminated"])
def test_failed_transition_requires_reset_before_another_step(monkeypatch, bad_field):
    import torch

    original = ToyPhysicsManager._step_model

    def invalid_step(cls, action):
        observation, reward, terminated, truncated, info = original(action)
        if bad_field == "observation":
            observation["latent"][:] = np.nan
        elif bad_field == "reward":
            reward[:] = np.nan
        else:
            terminated = terminated.astype(np.float32)
        return observation, reward, terminated, truncated, info

    with NeuralDirectEnv(make_generic_cfg()) as env:
        env.reset()
        with monkeypatch.context() as patch:
            patch.setattr(ToyPhysicsManager, "_step_model", classmethod(invalid_step))
            with pytest.raises(ValueError):
                env.step(torch.zeros(1, 2, 3))
            with pytest.raises(RuntimeError, match="Reset"):
                env.step(torch.zeros(1, 2, 3))
        env.reset()
        env.step(torch.zeros(1, 2, 3))
