# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Medical-pad unfold task: GR00T joint commands, Cosmos dynamics and image reward.

Isaac Lab owns simulation/reset. This module supplies the task-specific initial
conditions, H2 command codec and classifier. RLinf owns policy inference,
trajectory construction, optimization, checkpointing and evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from isaaclab_contrib.neural import NeuralDirectEnv
from isaaclab_contrib.neural.cosmos3 import Cosmos3InferenceCfg, make_cosmos3_env_cfg


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, dict):
        return {key: _numpy(item) for key, item in value.items()}
    return value


class _TaskEnvironment(NeuralDirectEnv):
    """Align task counters and initial states with native episode resets.

    With trainer-managed resets, keep terminal task data during native autoreset;
    chunk_step publishes the saved final observation until the next explicit reset.
    """

    def __init__(self, cfg, task):
        self.task = task
        super().__init__(cfg, render_mode="rgb_array")

    def _reset_idx(self, env_ids):
        if len(env_ids) and (not self.task._stepping or self.task.cfg.get("auto_reset", False)):
            self.task._prepare_reset(np.asarray(env_ids, dtype=np.int64))
            self.cfg.reset_options = copy.deepcopy(self.task.reset_options)
        super()._reset_idx(env_ids)


class Cosmos3MedicalPadUnfoldTask:
    """H2 medical-pad unfold task implementing RLinf's dense chunk contract.

    The reset manifest is a JSON list of records with ``image``, ``state`` (58
    measured joint values [m or rad, depending on joint type]), ``prompt`` and
    ``split``. Image paths resolve relative to the manifest. Live state after
    reset is explicitly the previous joint command, not a WM measurement.
    """

    def __init__(self, cfg: dict, *, num_envs: int, seed: int):
        from .classifier import MedicalPadClassifier
        from .h2_action_codec import H2UnifiedCodec

        self.cfg, self.num_envs, self.seed = cfg, num_envs, seed
        self._stepping = False  # Distinguish native autoreset during step from an explicit reset.
        task = cfg["isaaclab"]
        self.group_size = int(cfg.get("group_size", 1))
        if self.group_size < 1 or num_envs % self.group_size:
            raise ValueError("num_envs must be divisible by group_size")
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        assets = Path(__file__).with_name("assets")
        manifest = Path(task.get("reset_manifest") or assets / "resets.json").resolve()
        rows = json.loads(manifest.read_text())
        self.rows = []
        for row in rows:
            if row["split"] != task["split"]:
                continue
            row = dict(row)
            image = Path(row["image"])
            row["image"] = str(image if image.is_absolute() else manifest.parent / image)
            state = np.asarray(row["state"], dtype=np.float32)
            if state.shape != (58,) or not np.isfinite(state).all():
                raise ValueError("Each reset must provide 58 finite measured joint values")
            if not Path(row["image"]).is_file() or not row["prompt"].strip():
                raise ValueError("Reset image and prompt must exist")
            self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No reset records for split {task['split']!r}")
        model = Cosmos3InferenceCfg(**task["world_model"])
        model.output_dir = str(Path(model.output_dir) / f"{task['split']}-{seed}")
        self.chunk_size = model.chunk_size
        if self.chunk_size != 16 or model.mode != "window":
            raise ValueError("The H2 joint codec requires window inference with 16 actions per step")
        self.max_frames = int(cfg["max_episode_steps"])
        if self.max_frames % self.chunk_size:
            raise ValueError("max_episode_steps must be a multiple of the model chunk size")
        if cfg.get("ignore_terminations", False):
            raise ValueError("Classifier neural tasks require ignore_terminations=false")
        resources = Path(task.get("codec_resources") or assets / "resources")
        urdf = resources / "H2_with_sharpa_hands.urdf"
        self.codec = H2UnifiedCodec(urdf, hashlib.sha256(urdf.read_bytes()).hexdigest())
        reward = task["reward"]
        self.reward_device = torch.device(reward.get("device", "cuda"))
        classifier_config = reward.get("config_file") or assets / "classifier.yaml"
        self.classifier = MedicalPadClassifier(reward["checkpoint"], classifier_config, self.reward_device)
        self.success_class = int(reward.get("success_class", len(self.classifier.reward_values) - 1))
        self.failure_class = int(reward.get("failure_class", 0))
        if not 0 <= self.success_class < len(self.classifier.reward_values):
            raise ValueError("success_class is outside the classifier classes")
        if not 0 <= self.failure_class < len(self.classifier.reward_values) or self.failure_class == self.success_class:
            raise ValueError("failure_class must be a distinct classifier class")
        self.streak_frames = int(reward.get("terminal_streak_frames", 30))
        if self.streak_frames < 1:
            raise ValueError("terminal_streak_frames must be positive")
        self.timeout_success = bool(reward.get("timeout_success_if_final_success", True))
        self.states = np.zeros((num_envs, 58), np.float32)
        self.elapsed = np.zeros(num_envs, np.int64)
        self.previous = np.zeros(num_envs, np.float32)
        self.success_streak = np.zeros(num_envs, np.int64)
        self.failure_streak = np.zeros(num_envs, np.int64)
        self.returns = np.zeros(num_envs, np.float32)
        self.inactive = np.zeros(num_envs, bool)
        self.success = np.zeros(num_envs, bool)
        self.failure = np.zeros(num_envs, bool)
        self.reset_options = {}
        self.update_reset_state_ids()
        self._prepare_reset(np.arange(num_envs))

        def task_reward(frames, info):
            return self._reward(frames)

        env_cfg = make_cosmos3_env_cfg(
            model,
            initial_image=self.rows[0]["image"],
            prompt=self.rows[0]["prompt"],
            num_envs=num_envs,
            max_chunks=self.max_frames // self.chunk_size,
            seed=seed,
            reward_fn=task_reward,
        )
        self.env = _TaskEnvironment(env_cfg, self)
        self.last_obs = None

    def update_reset_state_ids(self) -> None:
        groups = self.num_envs // self.group_size
        if self.cfg["isaaclab"]["split"] == "train":
            choices = self.rng.integers(len(self.rows), size=groups)
        else:
            choices = (np.arange(groups) + self.epoch * groups) % len(self.rows)
        self.indices = np.repeat(choices, self.group_size)
        self.epoch += 1

    def _prepare_reset(self, ids):
        for index in ids:
            self.states[index] = self.rows[self.indices[index]]["state"]
        for values in (self.elapsed, self.previous, self.success_streak, self.failure_streak, self.returns):
            values[ids] = 0
        self.inactive[ids] = False
        self.success[ids] = self.failure[ids] = False
        self.reset_options = {
            "initial_images": [self.rows[i]["image"] for i in self.indices],
            "prompts": [self.rows[i]["prompt"] for i in self.indices],
            "seeds": [int((self.seed + self.epoch * 997 + i // self.group_size) % 2**32) for i in range(self.num_envs)],
        }

    def _obs(self, images, states):
        return {
            "main_images": np.asarray(images).copy(),
            "wrist_images": None,
            "states": np.asarray(states).copy(),
            "task_descriptions": list(self.reset_options["prompts"]),
        }

    def reset(self, seed: int | None = None, env_ids: np.ndarray | None = None) -> tuple[dict, dict]:
        """Reset selected rows; an explicit seed restarts deterministic reset sampling."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.epoch = 0
            self.update_reset_state_ids()
        if env_ids is None:
            obs, info = self.env.reset(seed=self.seed if seed is None else seed)
        else:
            self.env._reset_idx(np.asarray(env_ids, np.int64))
            obs, info = self.env._get_observations(), {}
        wrapped = self._obs(_numpy(obs["policy"]["main_images"]), self.states)
        self.last_obs = wrapped
        return wrapped, _numpy(info)

    def _reward(self, frames):
        with torch.inference_mode():
            scores, probabilities = self.classifier.score_chunk(
                torch.from_numpy(frames).permute(0, 1, 4, 2, 3).to(self.reward_device)
            )
        scores, labels = scores.cpu().numpy(), probabilities.argmax(-1).cpu().numpy()
        rewards = np.zeros((self.num_envs, self.chunk_size), np.float32)
        terminal = np.zeros_like(rewards, bool)
        truncated = np.zeros_like(rewards, bool)
        success = self.success.copy()
        failure = self.failure.copy()
        valid = np.zeros(self.num_envs, np.int64)
        for i in range(self.num_envs):
            if self.inactive[i]:
                terminal[i] = True
                continue
            for t in range(self.chunk_size):
                self.elapsed[i] += 1
                valid[i] += 1
                score = scores[i, t]
                rewards[i, t] = (score - self.previous[i]) * float(self.cfg.get("reward_coef", 1))
                self.previous[i] = score
                self.success_streak[i] = self.success_streak[i] + 1 if labels[i, t] == self.success_class else 0
                self.failure_streak[i] = self.failure_streak[i] + 1 if labels[i, t] == self.failure_class else 0
                success[i] = self.success_streak[i] >= self.streak_frames
                failure[i] = self.failure_streak[i] >= self.streak_frames
                timeout = self.elapsed[i] >= self.max_frames
                if timeout and self.timeout_success:
                    success[i] |= labels[i, t] == self.success_class
                if success[i] or failure[i] or timeout:
                    terminal[i, t:] = success[i] or failure[i]
                    truncated[i, t:] = timeout and not terminal[i, t]
                    break
        self.success, self.failure = success.copy(), failure.copy()
        self.returns += rewards.sum(axis=1)
        metrics = {
            "rewards": rewards,
            "terminations": terminal,
            "truncations": truncated,
            "valid_frames": valid,
            "success": success,
            "failure": failure,
            "return": self.returns.copy(),
            "episode_len": self.elapsed.copy(),
        }
        return rewards.sum(axis=1), terminal.any(axis=1), metrics

    def chunk_step(self, actions: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Advance one model chunk, retaining frame-level reward and done masks."""
        actions = np.asarray(actions, np.float32)
        if actions.shape != (self.num_envs, self.chunk_size, 58) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite H2 actions {(self.num_envs, self.chunk_size, 58)}")
        was_inactive = self.inactive.copy()
        controls = np.stack(
            [
                self.codec.for_inference(action, observed_start_joints=self.states[i])[:, :59]
                for i, action in enumerate(actions)
            ]
        )
        self._stepping = True
        try:
            obs, _, _, _, extras = self.env.step(torch.from_numpy(controls))
        finally:
            self._stepping = False
        transition = extras["neural_transition"]
        frames, metrics = transition["frames"], transition["task_metrics"]
        rewards, terms, truncs = (metrics[key] for key in ("rewards", "terminations", "truncations"))
        done = (terms | truncs).any(axis=1)
        final_states = actions[np.arange(self.num_envs), np.maximum(metrics["valid_frames"] - 1, 0)].copy()
        final_states[was_inactive] = self.states[was_inactive]
        final_obs = self._obs(frames[:, -1], final_states)
        if self.last_obs is not None:
            for key in ("main_images", "states"):
                final_obs[key][was_inactive] = self.last_obs[key][was_inactive]
        ongoing = ~done
        self.states[ongoing] = actions[ongoing, -1]
        result_obs = self._obs(_numpy(obs["policy"]["main_images"]), self.states)
        episode = {
            "success_once": metrics["success"],
            "fail_once": metrics["failure"],
            "return": metrics["return"],
            "episode_len": metrics["episode_len"],
        }
        info = {"episode": episode}
        if done.any():
            info.update(
                final_observation=final_obs,
                final_info={"episode": copy.deepcopy(episode)},
                _final_observation=done.copy(),
                _final_info=done.copy(),
            )
        if not self.cfg.get("auto_reset", False):
            for i in np.flatnonzero(done):
                result_obs["main_images"][i] = final_obs["main_images"][i]
                result_obs["states"][i] = final_states[i]
            if self.last_obs is not None:
                for key in ("main_images", "states"):
                    result_obs[key][was_inactive] = self.last_obs[key][was_inactive]
            self.states[done] = final_states[done]
            self.inactive |= done
        self.last_obs = result_obs
        observations = [self._obs(frames[:, t], actions[:, t]) for t in range(self.chunk_size)]
        for observation in observations:
            for key in ("main_images", "states"):
                observation[key][was_inactive] = result_obs[key][was_inactive]
        observations[-1] = result_obs
        return observations, rewards, terms, truncs, [info] * self.chunk_size

    def render(self) -> np.ndarray | None:
        """Render the visible task observation, including a frozen terminal row."""
        return self.env.render() if self.last_obs is None else self.last_obs["main_images"][0].copy()

    def close(self) -> None:
        if hasattr(self, "env"):
            self.env.close()


def create_env(cfg: dict, *, num_envs: int, seed: int) -> Cosmos3MedicalPadUnfoldTask:
    """Construct the task inside its isolated simulator process."""
    return Cosmos3MedicalPadUnfoldTask(cfg, num_envs=num_envs, seed=seed)
