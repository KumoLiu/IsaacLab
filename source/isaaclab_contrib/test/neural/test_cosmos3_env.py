# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native lifecycle and Cosmos I/O contracts, without downloading inference weights."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_contrib.neural.cosmos3 import Cosmos3InferenceCfg, Cosmos3PhysicsManager, make_cosmos3_env_cfg
from isaaclab_contrib.neural.cosmos3.simulator import Cosmos3Simulator
from isaaclab_contrib.neural.neural_env import NeuralDirectEnv


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "initial.png").touch()
    return Cosmos3InferenceCfg(checkpoint_path=str(tmp_path), output_dir=str(tmp_path / "out"), image_size=(16, 16))


class Contract:
    domain_ids = {"hand_pose": 3}

    def validate(self, payload, *, action_count, domain_name):
        assert payload["action_space"] == "normalized_unified_v1"
        values = np.asarray(payload["action"])
        if values.shape != (action_count, 59) or not np.isfinite(values).all():
            raise ValueError("invalid controls")
        return payload["action"], [self.domain_ids[domain_name]] * action_count


@pytest.fixture
def native_api(monkeypatch):
    """Replace only optional Cosmos inference; exercise actual Isaac Lab and simulator code."""
    calls = SimpleNamespace(loads=0, batches=[], generations=[], sessions=[], causal=False, input_size=(16, 16))

    def build_batch(**kwargs):
        calls.batches.append(copy.deepcopy(kwargs))
        height, width = kwargs["video"].shape[-2:]
        canvas = torch.nn.functional.pad(kwargs["video"], (0, 16 - width, 0, 16 - height))
        return {
            "video": [[canvas]],
            "action": [[kwargs["action"]]],
            "caption": [kwargs["prompt"]],
            "image_size": torch.tensor([[16, 16, height, width]]),
        }

    def generate(batch, **kwargs):
        calls.generations.append((copy.deepcopy(batch), kwargs))
        rgb = torch.zeros(1, 3, cfg_chunk[0] + 1, *calls.input_size)
        rgb[:, :, 1:] = 0.25
        return {"vision": [rgb]}

    cfg_chunk = [16]
    model = SimpleNamespace(
        config=SimpleNamespace(action_schema="unified_v1", max_action_dim=64, video_temporal_causal=False),
        input_video_key="video",
        tensor_kwargs={"device": "cpu"},
        _bimanual_action_contract=Contract(),
        generate_samples_from_batch=generate,
        decode=lambda rgb: rgb,
    )

    class Session:
        action_rows = 8
        closed = False
        observation = torch.zeros(1, 3, 1, 16, 16)

        def step(self, payload):
            self.payload = copy.deepcopy(payload)
            self.observation = torch.full((1, 3, 8, 16, 16), 0.75)
            return self.observation

        def close(self):
            self.closed = True

        def inspect(self):
            return {"closed": self.closed}

    def open_session(**kwargs):
        session = Session()
        session.request = kwargs
        calls.sessions.append(session)
        return session

    pipe = SimpleNamespace(model=model, open_bimanual_session=open_session)

    class Setup:
        def __init__(self, **kwargs):
            calls.setup = kwargs

        def build_setup(self):
            return self

        def get_inference_cls(self):
            return self

        def create(self, setup):
            calls.loads += 1
            return pipe

    modules = {
        "cosmos_framework.inference.args": {
            "OmniSetupOverrides": Setup,
            "ModelMode": SimpleNamespace(FORWARD_DYNAMICS="forward_dynamics"),
        },
        "cosmos_framework.inference.bimanual": {"BimanualActionContract": lambda *a: Contract()},
        "cosmos_framework.inference.action": {"build_action_batch": build_batch},
        "cosmos_framework.inference.vision": {
            "read_media_frames": lambda *a, **k: (torch.zeros(3, 1, *calls.input_size, dtype=torch.uint8), None)
        },
        "cosmos_framework.inference.inference": {"_finalize_data_batch": lambda batch, *a: batch},
        "cosmos_framework.data.generator.action.utils.domain_utils": {"get_domain_id": lambda name: 3},
    }
    for name, symbols in modules.items():
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            path = ".".join(parts[:i])
            if path not in sys.modules:
                module = ModuleType(path)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, path, module)
        for key, value in symbols.items():
            monkeypatch.setattr(sys.modules[name], key, value, raising=False)
    calls.model = model
    return calls


@pytest.mark.parametrize("training_flag", [None, "1"])
def test_window_generated_feedback_controls_and_independent_reset(cfg, native_api, monkeypatch, training_flag):
    if training_flag is None:
        monkeypatch.delenv("COSMOS_TRAINING", raising=False)
    else:
        monkeypatch.setenv("COSMOS_TRAINING", training_flag)
    model = Cosmos3Simulator(cfg, 2)
    model.load()
    assert os.environ["COSMOS_TRAINING"] == (training_flag or "0")
    assert native_api.setup["guardrails"] is False
    image = Path(cfg.checkpoint_path) / "initial.png"
    model.reset(np.array([0, 1]), [image, image], ["first task", "second task"], np.array([42, 99]))
    action = np.arange(2 * 16 * 59, dtype=np.float32).reshape(2, 16, 59) / 1000
    frames = model.step(action)
    assert frames.shape == (2, 16, 16, 16, 3) and frames.dtype == np.uint8
    assert all(
        torch.equal(batch[0]["action"][0][0][:, :59], torch.from_numpy(action[i]))
        for i, batch in enumerate(native_api.generations)
    )
    assert all(not batch[0]["action"][0][0][:, 59:].any() for batch in native_api.generations)
    assert native_api.batches[-1]["format_prompt_as_json"] is False
    assert [call[1]["seed"] for call in native_api.generations] == [[42], [99]]
    assert all(call[1]["shift"] == 10.0 for call in native_api.generations)
    model.reset(np.array([1]), [image, image], ["first task", "new task"], np.array([42, 101]))
    np.testing.assert_array_equal(model.info()["elapsed_frames"], [16, 0])
    model.step(action)
    assert [call[1]["seed"] for call in native_api.generations[-2:]] == [[58], [101]]
    # The next condition is the generated last frame, not another recorded image.
    assert torch.equal(native_api.batches[-2]["video"][:, 0], torch.from_numpy(frames[0, -1]).permute(2, 0, 1))
    assert native_api.loads == 1
    model.close()
    with pytest.raises(RuntimeError, match="Reset all"):
        model.step(action)


def test_session_preserves_controls_history_and_releases_on_reset(cfg, native_api):
    cfg = replace(cfg, mode="session", chunk_size=8)
    model = Cosmos3Simulator(cfg, 1)
    model.load()
    image = Path(cfg.checkpoint_path) / "initial.png"
    model.reset(np.array([0]), [image], ["task"], np.array([7]))
    actions = np.full((1, 8, 59), 0.125, np.float32)
    frames = model.step(actions)
    assert frames.shape == (1, 8, 16, 16, 3)
    assert native_api.sessions[0].payload == {"action_space": "normalized_unified_v1", "action": actions[0].tolist()}
    np.testing.assert_allclose(model.info()["elapsed_seconds"], [8 / 30])
    model.reset(np.array([0]), [image], ["another task"], np.array([8]))
    assert native_api.sessions[0].closed and native_api.loads == 1
    assert native_api.sessions[1].request["prompt"] == "another task"
    model.close()
    assert native_api.sessions[1].closed


def test_causal_checkpoint_cannot_silently_use_window_sampling(cfg, native_api):
    native_api.model.config.video_temporal_causal = True
    model = Cosmos3Simulator(cfg, 1)
    with pytest.raises(ValueError, match="mode='session'"):
        model.load()
    model.close()


@pytest.mark.parametrize(
    "updates",
    [
        {"chunk_size": 3},
        {"fps": 0},
        {"chunk_size": True},
        {"num_inference_steps": 0},
        {"mode": "other"},
        {"image_size": (0, 16)},
        {"guidance": float("nan")},
        {"mode": "session", "num_inference_steps": 4},
    ],
)
def test_invalid_config_fails_before_loading(cfg, updates):
    with pytest.raises(ValueError):
        replace(cfg, **updates).validate()


def test_session_rejects_multiple_environments_before_loading(cfg):
    with pytest.raises(ValueError, match="one environment"):
        make_cosmos3_env_cfg(
            replace(cfg, mode="session", chunk_size=8),
            initial_image=str(Path(cfg.checkpoint_path) / "initial.png"),
            prompt="task",
            num_envs=2,
        )


def test_native_autoreset_preserves_terminal_pixels_and_reward(cfg, native_api):
    def reward(frames, info):
        return np.array([2.0], np.float32), np.array([True]), {"success": True}

    native_cfg = make_cosmos3_env_cfg(
        cfg, initial_image=str(Path(cfg.checkpoint_path) / "initial.png"), prompt="task", reward_fn=reward
    )
    with NeuralDirectEnv(native_cfg, render_mode="rgb_array") as env:
        obs, _ = env.reset(seed=123)
        assert obs["policy"]["main_images"].shape == (1, 16, 16, 3)
        assert env.render().max() == 0
        assert not native_api.generations
        obs, reward_value, terminated, truncated, extras = env.step(torch.zeros(1, 16, 59))
        assert reward_value.item() == 2 and terminated.item() and not truncated.item()
        assert obs["policy"]["main_images"].max() == 0
        assert extras["final_obs"]["policy"]["main_images"].max() > 0
        assert extras["neural_transition"]["frames"].shape == (1, 16, 16, 16, 3)
        assert extras["neural_transition"]["task_metrics"]["success"] is True
        assert native_api.loads == 1
    assert Cosmos3PhysicsManager._model is None


def test_native_horizon_and_partial_reset(cfg, native_api):
    native_cfg = make_cosmos3_env_cfg(
        cfg, initial_image=str(Path(cfg.checkpoint_path) / "initial.png"), prompt="task", num_envs=2, max_chunks=2
    )
    with NeuralDirectEnv(native_cfg) as env:
        env.reset(seed=23)
        _, reward, done, truncated, _ = env.step(torch.zeros(2, 16, 59))
        assert not done.any() and not truncated.any() and not reward.any()
        env.sim.physics_manager.reset_episode(env_ids=np.array([1]), seed=90, options=native_cfg.reset_options)
        assert env.sim.physics_manager.observation["main_images"][0].max() > 0
        assert env.sim.physics_manager.observation["main_images"][1].max() == 0
        _, _, _, truncated, _ = env.step(torch.zeros(2, 16, 59))
        assert truncated.all()


def test_task_registration_and_config_do_not_import_cosmos_or_rlinf():
    script = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in ("cosmos_framework", "rlinf"):
            raise AssertionError("Optional inference library imported during registration")
sys.meta_path.insert(0, Block())
import gymnasium as gym
import isaaclab_tasks.contrib.cosmos3_medical_pad_unfold
from isaaclab_contrib.neural.cosmos3 import Cosmos3InferenceCfg
assert gym.spec("Isaac-H2-MedicalPad-Unfold-Cosmos3-Direct-v0").entry_point.endswith(":NeuralDirectEnv")
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_missing_framework_explains_optional_install(cfg):
    script = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname == "cosmos_framework":
            raise ModuleNotFoundError("missing optional Framework", name=fullname)
sys.meta_path.insert(0, Block())
from isaaclab_contrib.neural.cosmos3 import Cosmos3InferenceCfg
from isaaclab_contrib.neural.cosmos3.simulator import Cosmos3Simulator
cfg = Cosmos3InferenceCfg(checkpoint_path=sys.argv[1], output_dir=sys.argv[2])
try:
    Cosmos3Simulator(cfg, 1).load()
except ModuleNotFoundError as error:
    assert "active Isaac Lab environment" in str(error)
    assert "uv sync --extra cosmos3" in str(error)
else:
    raise AssertionError("Missing optional SDK must fail before model loading")
"""
    subprocess.run(
        [sys.executable, "-c", script, cfg.checkpoint_path, cfg.output_dir],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("override", ["image_size", "reward_fn"])
def test_changed_contract_is_rejected_before_model_load(cfg, override):
    native_cfg = make_cosmos3_env_cfg(cfg, initial_image=str(Path(cfg.checkpoint_path) / "initial.png"), prompt="task")
    if override == "image_size":
        native_cfg.sim.physics.inference.image_size = (32, 32)
    else:
        native_cfg.sim.physics.reward_fn = 1
    with pytest.raises(ValueError, match="spaces must match|reward_fn must be callable"):
        Cosmos3PhysicsManager.validate_env_cfg(native_cfg)


def test_chunk_horizon_does_not_round_up_an_extra_step(cfg, native_api):
    native_cfg = make_cosmos3_env_cfg(
        cfg, initial_image=str(Path(cfg.checkpoint_path) / "initial.png"), prompt="task", max_chunks=125
    )
    with NeuralDirectEnv(native_cfg) as env:
        assert env.max_episode_length == 125


def test_failed_startup_releases_model_and_allows_a_fresh_environment(cfg, native_api):
    native_cfg = make_cosmos3_env_cfg(cfg, initial_image=str(Path(cfg.checkpoint_path) / "initial.png"), prompt="task")
    native_api.model.config.action_schema = "native"
    with pytest.raises(ValueError, match="unified_v1"):
        with NeuralDirectEnv(native_cfg) as env:
            env.reset()
    assert Cosmos3PhysicsManager._model is None
    native_api.model.config.action_schema = "unified_v1"
    with NeuralDirectEnv(native_cfg) as env:
        env.reset()
    assert native_api.loads == 2


def test_window_feedback_preserves_native_content_rectangle(cfg, native_api):
    native_api.input_size = (12, 10)
    cfg = replace(cfg, image_size=(12, 10))
    model = Cosmos3Simulator(cfg, 1)
    model.load()
    image = Path(cfg.checkpoint_path) / "initial.png"
    model.reset(np.array([0]), [image], ["task"], np.array([42]))
    frames = model.step(np.zeros((1, 16, 59), np.float32))
    # Padding is model canvas, not additional observed scene content.
    assert native_api.batches[-1]["video"].shape[-2:] == (12, 10)
    model.step(np.zeros((1, 16, 59), np.float32))
    torch.testing.assert_close(
        native_api.batches[-1]["video"][:, 0], torch.from_numpy(frames[0, -1, :12, :10]).permute(2, 0, 1)
    )
    model.close()


def test_rgb_quantization_matches_native_public_api(cfg, native_api):
    model = Cosmos3Simulator(cfg, 1)
    model.load()
    image = Path(cfg.checkpoint_path) / "initial.png"
    model.reset(np.array([0]), [image], ["task"], np.array([42]))
    # Native video postprocessing casts 0.5 * 255 to uint8 (127), without rounding.
    native_api.model.decode = lambda _: torch.zeros(1, 3, 17, 16, 16)
    frames = model.step(np.zeros((1, 16, 59), np.float32))
    assert np.all(frames == 127)
    model.close()
