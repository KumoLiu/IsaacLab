# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural worker isolation and the RLinf chunk interface, without model weights."""

import multiprocessing as mp

import pytest
import torch
from omegaconf import OmegaConf

from isaaclab_contrib.rl.rlinf.neural_env import NeuralRlinfEnv


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    module = tmp_path / "neural_test_factory.py"
    module.write_text("""import numpy as np
class Task:
    def __init__(self, cfg, num_envs, seed):
        self.n = num_envs
        self.calls = 0
    def reset(self, seed=None, env_ids=None):
        return {"main_images": np.zeros((self.n, 2, 2, 3), np.uint8)}, {}
    def chunk_step(self, actions):
        if not np.isfinite(actions).all():
            raise ValueError("non-finite task action")
        self.calls += 1
        rewards = np.ones(actions.shape[:2], np.float32) * self.calls
        done = np.zeros(actions.shape[:2], bool)
        info = {"episode": {"episode_len": np.full(self.n, actions.shape[1], np.int64)}}
        return [{"states": actions[:, -1]}], rewards, done, done, [info]
    def close(self): pass
    def update_reset_state_ids(self): pass
    def render(self): return np.zeros((2, 2, 3), np.uint8)
def create(cfg, *, num_envs, seed):
    if cfg.get("fail_start"):
        raise ValueError("task startup failed")
    return Task(cfg, num_envs, seed)
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    return OmegaConf.create(
        {"seed": 3, "auto_reset": False, "isaaclab": {"env_factory": "neural_test_factory:create", "timeout_s": 20}}
    )


def test_chunk_is_one_simulator_call(cfg):
    env = NeuralRlinfEnv(cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None)
    process = env._process
    try:
        obs, _ = env.reset()
        assert obs["main_images"].dtype == torch.uint8
        result = env.chunk_step(torch.ones(2, 16, 3))
        assert result[1].shape == (2, 16)
        torch.testing.assert_close(result[1], torch.ones(2, 16))
        torch.testing.assert_close(env.elapsed_steps, torch.full((2,), 16, dtype=torch.int32))
        assert process.pid != mp.current_process().pid
    finally:
        env.close()
    assert not process.is_alive()
    env.close()


def test_child_failure_reaches_driver(cfg):
    env = NeuralRlinfEnv(cfg, 1, 0, 1, None)
    try:
        with pytest.raises(RuntimeError, match="non-finite task action"):
            env.chunk_step(torch.full((1, 16, 3), float("nan")))
    finally:
        env.close()
    assert not env._process.is_alive()


def test_startup_failure_does_not_hang(cfg):
    cfg.fail_start = True
    with pytest.raises(RuntimeError, match="task startup failed"):
        NeuralRlinfEnv(cfg, 1, 0, 1, None)
