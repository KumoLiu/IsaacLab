# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf environment interface for task-owned Isaac Lab neural simulations.

Each instance owns a simulator subprocess. Train/eval environments therefore do
not compete for Isaac Lab's process-local SimulationContext singleton. Task
factories supply reset/chunk_step; RLinf retains rollout and training ownership.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def _resolve(entry_point: str):
    module, name = entry_point.split(":", 1)
    return getattr(importlib.import_module(module), name)


def _worker(connection: Connection, factory: str, cfg: dict, num_envs: int, seed: int) -> None:
    env = None
    try:
        env = _resolve(factory)(cfg, num_envs=num_envs, seed=seed)
        connection.send((True, None))
        while True:
            method, args, kwargs = connection.recv()
            if method == "close":
                break
            connection.send((True, getattr(env, method)(*args, **kwargs)))
    except EOFError:
        pass
    except BaseException:
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            connection.send((False, traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        connection.close()


def _tensors(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: _tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tensors(item) for item in value)
    return value


class NeuralRlinfEnv(gym.Env):
    """Run a configured neural task behind RLinf's reset/chunk_step contract.

    ``cfg.isaaclab.env_factory`` is a ``module:callable`` accepting a plain
    configuration dictionary, ``num_envs`` and ``seed``. Its result exposes
    reset, chunk_step, update_reset_state_ids and close. Chunk transitions use
    the RLinf contract, with rewards and done flags shaped ``(N, C)``.
    """

    def __init__(self, cfg: DictConfig, num_envs: int, seed_offset: int, total_num_processes: int, worker_info: object):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = cfg.seed + seed_offset
        self.device = torch.device("cpu")
        self._is_start = True
        self._closed = False
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.int32)
        self._timeout = float(cfg.isaaclab.get("timeout_s", 600))
        if self._timeout <= 0:
            raise ValueError("isaaclab.timeout_s must be positive")
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(
            target=_worker,
            args=(child, cfg.isaaclab.env_factory, OmegaConf.to_container(cfg, resolve=True), num_envs, self.seed),
            daemon=True,
        )
        self._process.start()
        child.close()
        try:
            self._receive()
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _receive(self):
        if not self._connection.poll(self._timeout):
            raise TimeoutError(f"Neural simulator did not respond within {self._timeout:g} seconds")
        try:
            success, result = self._connection.recv()
        except EOFError as error:
            raise RuntimeError(f"Neural simulator exited with code {self._process.exitcode}") from error
        if not success:
            raise RuntimeError(f"Neural simulator failed:\n{result}")
        return _tensors(result)

    def _call(self, method: str, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Neural environment is closed")
        self._connection.send((method, args, kwargs))
        return self._receive()

    def reset(self, seed: int | None = None, env_ids: torch.Tensor | None = None) -> tuple[dict, dict]:
        ids = None if env_ids is None else torch.as_tensor(env_ids).cpu().numpy()
        result = self._call("reset", seed=seed, env_ids=ids)
        if ids is None:
            self._elapsed_steps.zero_()
        else:
            self._elapsed_steps[ids] = 0
        return result

    def chunk_step(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        controls = torch.as_tensor(actions).detach().cpu().numpy()
        result = self._call("chunk_step", controls)
        info = result[-1][-1]
        self._elapsed_steps = torch.as_tensor(info["episode"]["episode_len"], dtype=torch.int32).clone()
        if self.cfg.get("auto_reset", False):
            done = (result[2] | result[3]).any(dim=1)
            self._elapsed_steps[done] = 0
        return result

    def update_reset_state_ids(self) -> None:
        self._call("update_reset_state_ids")

    def render(self) -> np.ndarray | None:
        result = self._call("render")
        return result.numpy() if isinstance(result, torch.Tensor) else result

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps.clone()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                self._connection.send(("close", (), {}))
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=10)
        self._connection.close()
        atexit.unregister(self.close)
