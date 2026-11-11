# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classifier termination and dense rewards across model chunk boundaries."""

import numpy as np
import torch

from isaaclab_tasks.contrib.cosmos3_medical_pad_unfold.rlinf_env import Cosmos3MedicalPadUnfoldTask


class Classifier:
    label = 4

    def score_chunk(self, frames):
        n, t = frames.shape[:2]
        labels = torch.full((n, t), self.label, dtype=torch.long)
        probabilities = torch.nn.functional.one_hot(labels, 5).float()
        scores = (probabilities * torch.tensor([-2, -1, 0, 1, 2])).sum(-1)
        return scores, probabilities


def task(max_frames=480):
    env = object.__new__(Cosmos3MedicalPadUnfoldTask)
    env.num_envs, env.chunk_size, env.max_frames = 1, 16, max_frames
    env.cfg = {"reward_coef": 1}
    env.reward_device = torch.device("cpu")
    env.classifier = Classifier()
    env.streak_frames, env.success_class, env.failure_class = 30, 4, 0
    env.timeout_success = True
    for name in ("elapsed", "success_streak", "failure_streak"):
        setattr(env, name, np.zeros(1, np.int64))
    for name in ("previous", "returns"):
        setattr(env, name, np.zeros(1, np.float32))
    for name in ("inactive", "success", "failure"):
        setattr(env, name, np.zeros(1, bool))
    return env


def test_success_streak_spans_chunks_and_masks_terminal_tail():
    env = task()
    frames = np.zeros((1, 16, 2, 2, 3), np.uint8)
    _, done, _ = env._reward(frames)
    assert not done.any()
    _, done, metrics = env._reward(frames)
    assert done.tolist() == [True]
    assert metrics["valid_frames"].tolist() == [14]
    assert metrics["terminations"][0].tolist() == [False] * 13 + [True] * 3
    np.testing.assert_array_equal(metrics["rewards"][0, 14:], 0)
    env.inactive[:] = True
    _, _, after = env._reward(frames)
    assert after["valid_frames"].tolist() == [0]
    assert after["success"].tolist() == [True]
    np.testing.assert_array_equal(after["rewards"], 0)
    np.testing.assert_array_equal(after["return"], [2])


def test_timeout_does_not_turn_progress_into_success():
    env = task(max_frames=32)
    env.classifier.label = 3
    frames = np.zeros((1, 16, 2, 2, 3), np.uint8)
    env._reward(frames)
    _, _, metrics = env._reward(frames)
    assert not metrics["success"].any()
    assert not metrics["terminations"].any()
    assert metrics["truncations"][0, -1]
    assert metrics["episode_len"].tolist() == [32]
