# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Native Isaac Lab boundary: grouped resets, command rewards and terminal frames."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions_for_isaaclab

from isaaclab_tasks.contrib.dreamdojo_trocar.rlinf_adapter import TrocarRLinfEnv


@pytest.mark.parametrize("with_app", [False, True])
def test_subproc_isaaclab_native_and_legacy_factory(tmp_path, with_app):
    from rlinf.envs.isaaclab.venv import SubProcIsaacLabEnv

    def factory():
        class Env:
            device = torch.device("cpu")

            def reset(self, seed=None, env_ids=None, options=None):
                return torch.tensor([seed]), {"env_ids": env_ids, "options": options}

            def step(self, action):
                return action + 1, 1.0, False, False, {}

            def close(self):
                (tmp_path / "env_closed").touch()

        class App:
            def close(self):
                (tmp_path / "app_closed").touch()

        return (Env(), App()) if with_app else Env()

    env = SubProcIsaacLabEnv(factory) if with_app else SubProcIsaacLabEnv(factory, timeout_s=30)
    process = env.isaac_lab_process
    try:
        obs, info = env.reset(7, torch.tensor([0]))
        assert obs.tolist() == [7] and info["env_ids"].tolist() == [0]
        obs, info = env.reset(seed=8, options={"episode_index": 2})
        assert info["options"] == {"episode_index": 2}
        assert env.device() == torch.device("cpu")
        assert env.step(torch.tensor([[3.0]]))[0].item() == 4
    finally:
        env.close()
        env.close()
    assert not process.is_alive() and (tmp_path / "env_closed").is_file()
    assert (tmp_path / "app_closed").is_file() == with_app


def test_subproc_isaaclab_startup_error_reaches_parent():
    from rlinf.envs.isaaclab.venv import SubProcIsaacLabEnv

    def factory():
        raise ValueError("invalid model checkpoint")

    with pytest.raises(RuntimeError, match="invalid model checkpoint"):
        SubProcIsaacLabEnv(factory, timeout_s=30)


@pytest.mark.parametrize("failure", ["exception", "exit", "timeout"])
def test_subproc_isaaclab_step_failure_closes_child(failure):
    from rlinf.envs.isaaclab.venv import SubProcIsaacLabEnv

    def factory():
        import torch

        class Env:
            device = torch.device("cpu")

            def step(self, action):
                if failure == "exception":
                    raise ValueError("invalid action")
                if failure == "exit":
                    import os

                    os._exit(7)
                import time

                time.sleep(30)

            def close(self):
                pass

        return Env()

    env = SubProcIsaacLabEnv(factory, timeout_s=30)
    if failure == "timeout":
        env.timeout_s = 0.1
    try:
        expected = {"exception": "invalid action", "exit": "exited: 7", "timeout": "timed out"}[failure]
        with pytest.raises((RuntimeError, TimeoutError), match=expected):
            env.step(torch.zeros(1))
        assert not env.isaac_lab_process.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            env.step(torch.zeros(1))
    finally:
        env.close()


@pytest.fixture
def neural_recipe_path():
    from isaaclab_rl.entrypoints.backends.cli_args_rlinf import resolve_config_dir

    name = "isaaclab_neural_gr00t"
    path = Path(resolve_config_dir(name, None)) / f"{name}.yaml"
    assert path.is_file()
    return path


@pytest.fixture
def native_cfg(tmp_path, monkeypatch):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"total_episodes": 3}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-uuid")
    from isaaclab_tasks.contrib.dreamdojo_trocar import rlinf_adapter as task_adapter

    class FakeNative:
        def __init__(self, factory, **kwargs):
            self.requests, self.closed, self.count = [], False, 0
            assert callable(factory)

        def reset(self, *, seed, options):
            self.num_envs = len(options["episode_indices"])
            arrays, info = self.request("reset", seed=seed, **options)
            return self.observation(arrays), {"reset_info": {**info, "frames": arrays["frames"]}}

        @staticmethod
        def observation(arrays):
            return {"policy": {k: torch.from_numpy(arrays[k]) for k in ("main_images", "states")}}

        def step(self, action):
            arrays, info = self.request("step", action=action)
            obs = self.observation(arrays)
            return (
                obs,
                torch.from_numpy(arrays["command_rewards"].sum(axis=1)),
                torch.zeros(self.num_envs, dtype=torch.bool),
                torch.full((self.num_envs,), arrays["truncated"]),
                {"neural_transition": {**info, **arrays}, "final_obs": obs},
            )

        def request(self, command, **kwargs):
            self.requests.append((command, kwargs))
            self.count = 0 if command == "reset" else self.count + 1
            value = self.count
            arrays = {
                "main_images": np.full((self.num_envs, 2, 2, 3), value, np.uint8),
                "states": np.zeros((self.num_envs, 28), np.float32),
                "frames": np.full((self.num_envs, 1 if command == "reset" else 6, 2, 2, 3), value, np.uint8),
                "command_rewards": np.tile(np.array([0, 1] + [0] * 10, np.float32), (self.num_envs, 1)),
                "probabilities": np.zeros((self.num_envs, 6, 3), np.float32),
                "terminated": False,
                "truncated": self.count == 2,
            }
            return arrays, {
                "task_description": ["Pick, hand over, place."] * self.num_envs,
                "milestone_stage": np.full(self.num_envs, min(value, 1)),
                "is_success": np.zeros(self.num_envs, dtype=bool),
            }

        def close(self):
            self.closed = True

    monkeypatch.setattr(task_adapter, "SubProcIsaacLabEnv", FakeNative)
    return OmegaConf.create(
        {
            "env_type": "isaaclab",
            "isaaclab": {
                "backend": "neural",
                "adapter": "isaaclab_tasks.contrib.dreamdojo_trocar.rlinf_adapter:TrocarRLinfEnv",
            },
            "init_params": {"id": "Isaac-DreamDojo-Trocar-Direct-v0"},
            "seed": 1234,
            "group_size": 2,
            "auto_reset": False,
            "ignore_terminations": False,
            "is_eval": False,
            "enable_offload": False,
            "use_fixed_reset_state_ids": False,
            "reset_episode_ids": [0, 1],
            "max_episode_steps": 24,
            "max_steps_per_rollout_epoch": 24,
            "reward_coef": 2.0,
            "native_log_dir": str(tmp_path),
            "inference": {
                "max_chunks": 2,
                "dataset": str(tmp_path),
                **{
                    k: str(tmp_path)
                    for k in (
                        "dreamdojo_root",
                        "wm_checkpoint",
                        "reward_checkpoint",
                        "lam_checkpoint",
                    )
                },
            },
        }
    )


def test_grouped_resets_rewards_and_terminal_observations(native_cfg):
    with TrocarRLinfEnv(native_cfg, 2, 0, 1, None) as env:
        with pytest.raises(RuntimeError, match="Reset"):
            env.chunk_step(torch.zeros(2, 12, 28))
        obs, _ = env.reset()
        assert obs["main_images"].max() == 0
        request = env._worker.requests[-1][1]
        assert request == {"episode_indices": [0, 0], "seeds": [1234, 1234], "seed": 1234}
        actions = np.full((2, 12, 28), 2.5, np.float32)
        decoded = prepare_actions_for_isaaclab(actions, "gr00t_n1d7")
        torch.testing.assert_close(decoded, torch.from_numpy(actions))
        obs, rewards, terms, truncs, infos = env.chunk_step(decoded)
        assert rewards.shape == (2, 12) and rewards[:, 1].tolist() == [2.0, 2.0]
        assert not terms.any() and not truncs.any()
        assert infos[0]["episode"]["picked"].all() and not infos[0]["episode"]["success_once"].any()
        obs, _, terms, truncs, infos = env.chunk_step(decoded)
        assert obs[0]["main_images"].max() == 2  # Terminal image, not native reset image.
        assert truncs[:, -1].all() and not truncs[:, :-1].any() and not terms.any()
        assert "final_observation" not in infos[0]  # No fake auto-reset in RLinf.
        assert env.capture_image().shape == (2, 6, 2, 2, 3)
        from rlinf.envs.wrappers.record_video import RecordVideo

        recorder = RecordVideo(env, OmegaConf.create({"fps": 15}))
        try:
            assert len(recorder._extract_frame_batches(obs)) == 6
        finally:
            recorder._executor.shutdown()
        assert infos[0]["episode"]["episode_len"].tolist() == [24, 24]
        with pytest.raises(RuntimeError, match="Reset"):
            env.chunk_step(decoded)
        env.update_reset_state_ids()
        env.reset()
        assert env._worker.requests[-1][1] == {"episode_indices": [1, 1], "seeds": [1235, 1235], "seed": 1234}
        env.reset(seed=1234)
        assert env._worker.requests[-1][1] == request
        worker = env._worker
    assert worker.closed


@pytest.mark.parametrize(
    "key,value",
    [
        ("auto_reset", True),
        ("ignore_terminations", True),
        ("enable_offload", True),
        ("max_episode_steps", 13),
        ("reset_episode_ids", [999]),
    ],
)
def test_unsupported_native_settings_fail_before_start(native_cfg, key, value):
    native_cfg[key] = value
    with pytest.raises(ValueError):
        TrocarRLinfEnv(native_cfg, 2, 0, 1, None)


def test_native_fixed_cases_and_invalid_actions(native_cfg):
    native_cfg.use_fixed_reset_state_ids = True
    with TrocarRLinfEnv(native_cfg, 2, 1, 2, None) as env:
        env.reset()
        first = env._worker.requests[-1]
        assert first[1] == {"episode_indices": [1, 1], "seeds": [1235, 1235], "seed": 1234}
        for action in (torch.zeros(2, 28), torch.full((2, 12, 28), float("nan"))):
            with pytest.raises(ValueError, match="absolute joint"):
                env.chunk_step(action)
        assert len(env._worker.requests) == 1
        env.update_reset_state_ids()
        env.reset()
        assert env._worker.requests[-1] == first


def test_native_request_failure_closes_shared_worker(native_cfg, monkeypatch):
    with TrocarRLinfEnv(native_cfg, 2, 0, 1, None) as env:
        env.reset()
        worker = env._worker

        def fail(*args, **kwargs):
            raise TimeoutError("inference failed")

        monkeypatch.setattr(worker, "request", fail)
        with pytest.raises(TimeoutError):
            env.chunk_step(torch.zeros(2, 12, 28))
        assert worker.closed


def test_native_recipe_builds_valid_grpo_advantages(neural_recipe_path):
    """The actual actor must build a mask, even with fixed-horizon episodes."""
    from rlinf.algorithms.registry import calculate_adv_and_returns
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

    recipe = OmegaConf.load(neural_recipe_path)
    rewards = torch.zeros(2, 2, 12)
    rewards[-1, 1, -1] = 1.0  # Different outcome within the same group.
    dones = torch.zeros(3, 2, 12, dtype=torch.bool)
    dones[-1, :, -1] = True
    batch = EmbodiedFSDPActor._process_received_rollout_batch(
        SimpleNamespace(cfg=recipe), {"rewards": rewards, "dones": dones}
    )
    assert batch["loss_mask"].shape == rewards.shape
    assert batch["loss_mask"].all()  # Include the last command's reward.
    assert batch["loss_mask_sum"].unique().tolist() == [24]
    advantages = calculate_adv_and_returns(
        task_type=recipe.runner.task_type,
        adv_type=recipe.algorithm.adv_type,
        reward_type=recipe.algorithm.reward_type,
        group_size=recipe.algorithm.group_size,
        **batch,
    )["advantages"]
    assert advantages.shape == rewards.shape and torch.isfinite(advantages).all()
    assert (advantages[:, 0] < 0).all() and (advantages[:, 1] > 0).all()


def test_native_recipe_loads_and_registers_without_published_config(neural_recipe_path, tmp_path, monkeypatch):
    import rlinf
    from hydra import compose, initialize_config_dir
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS
    from rlinf.models.embodiment.gr00t import simulation_io

    from isaaclab_contrib.rl.rlinf import extension

    from isaaclab_rl.entrypoints.backends.cli_args_rlinf import resolve_config_dir

    root = Path(__file__).resolve().parents[4]
    monkeypatch.chdir(root)
    assert resolve_config_dir(neural_recipe_path.stem, str(neural_recipe_path.parent.relative_to(root))) == str(
        neural_recipe_path.parent
    )
    monkeypatch.setenv("RLINF_ROOT", str(Path(rlinf.__file__).resolve().parents[1]))
    monkeypatch.setenv("NEURAL_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DREAMDOJO_LAM_CHECKPOINT", str(tmp_path / "LAM_400k.ckpt"))
    with initialize_config_dir(config_dir=str(neural_recipe_path.parent), version_base="1.1"):
        cfg = compose(config_name=neural_recipe_path.stem)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["actor"]["model"]["precision"] == "bf16"  # From RLinf's model defaults.
    assert resolved["actor"]["model"]["action_dim"] == 28  # Task-specific override.
    assert resolved["rollout"]["model"] == resolved["actor"]["model"]

    monkeypatch.setenv("RLINF_CONFIG_FILE", str(neural_recipe_path))
    monkeypatch.delitem(REGISTER_ISAACLAB_ENVS, cfg.env.train.init_params.id, raising=False)
    monkeypatch.setattr(extension, "_full_cfg_cache", None)
    monkeypatch.setattr(extension, "_registered", False)
    monkeypatch.delitem(simulation_io.OBS_CONVERSION, "g1_dex3_wm", raising=False)
    monkeypatch.delitem(simulation_io.ACTION_CONVERSION_N1D7, "g1_dex3_wm", raising=False)

    def forbidden(*args):
        raise AssertionError("Neural tasks must not patch the policy implementation")

    monkeypatch.setattr(extension, "_register_gr00t_converters", forbidden)
    monkeypatch.setattr(extension, "_patch_gr00t_get_model", forbidden)
    extension.register()
    assert get_env_cls("isaaclab", cfg.env.train) is TrocarRLinfEnv
    assert get_env_cls("isaaclab", cfg.env.eval) is TrocarRLinfEnv
    assert extension._load_full_cfg()["env"]["train"]["init_params"]["id"] == cfg.env.train.init_params.id
    state = torch.arange(56, dtype=torch.float32).reshape(2, 28)
    obs = simulation_io.OBS_CONVERSION["g1_dex3_wm"](
        {
            "main_images": torch.zeros(2, 8, 8, 3, dtype=torch.uint8),
            "states": state,
            "task_descriptions": ["pick", "place"],
        }
    )
    assert obs["video.head_view"].shape == (2, 1, 8, 8, 3)
    assert obs["annotation.human.task_description"] == ["pick", "place"]
    groups = ("left_arm", "right_arm", "left_hand", "right_hand")
    for index, name in enumerate(groups):
        np.testing.assert_array_equal(obs[f"state.{name}"], state[:, None, 7 * index : 7 * (index + 1)])
    # Dictionary insertion order must not change physical joint ordering.
    parts = {name: np.full((2, 16, 7), float(i + 2)) for i, name in reversed(list(enumerate(groups)))}
    converter = simulation_io.ACTION_CONVERSION_N1D7["g1_dex3_wm"]
    expected = np.concatenate([parts[name][:, :12] for name in groups], axis=-1)
    for payload in (parts, {f"action.{k}": v for k, v in parts.items()}):
        np.testing.assert_array_equal(converter(payload, 12), expected)


def test_success_is_independent_of_episode_termination(native_cfg, monkeypatch):
    with TrocarRLinfEnv(native_cfg, 2, 0, 1, None) as env:
        env.reset()
        original = env._worker.request

        def successful(*args, **kwargs):
            arrays, info = original(*args, **kwargs)
            info.update(is_success=np.array([True, False]), milestone_stage=np.array([3, 1]))
            return arrays, info

        monkeypatch.setattr(env._worker, "request", successful)
        _, _, terms, truncs, infos = env.chunk_step(torch.zeros(2, 12, 28))
        assert infos[0]["episode"]["success_once"].tolist() == [True, False]
        assert not terms.any() and not truncs.any()


def test_multiple_groups_share_one_worker_but_not_cases_or_seeds(native_cfg):
    with TrocarRLinfEnv(native_cfg, 4, 0, 1, None) as env:
        env.reset()
        assert env._worker.requests[-1][1] == {
            "episode_indices": [0, 0, 1, 1],
            "seeds": [1234, 1234, 1235, 1235],
            "seed": 1234,
        }
        obs, _, _, _, _ = env.chunk_step(torch.zeros(4, 12, 28))
        assert obs[0]["states"].shape == (4, 28)
        assert len(env._worker.requests) == 2  # One reset and one batched step.
