# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run real DreamDojo inference through the robot-free Isaac Lab contribution.

Replay recorded actions; generated images feed the next WM step. No future real
image/state is injected after reset. No robot or policy is loaded. Use RLinf's
native train/play entrypoints for policy rollouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext

from isaaclab_contrib.neural.neural_env import NeuralDirectEnv

from isaaclab_tasks.contrib.dreamdojo_trocar.dataset import load_recording
from isaaclab_tasks.contrib.dreamdojo_trocar.env_cfg import (
    DreamDojoInferenceCfg,
    DreamDojoPhysicsManager,
    make_dreamdojo_env_cfg,
)


def comparison_frame(real: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """Label reference and generated images without changing model inputs."""
    canvas = Image.new("RGB", (1280, 512), "#152031")
    canvas.paste(Image.fromarray(real), (0, 32))
    canvas.paste(Image.fromarray(generated), (640, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "REAL RECORDING | reference only", fill="white")
    draw.text((652, 8), "DREAMDOJO | recorded actions, generated-image feedback", fill="white")
    return np.asarray(canvas)


def main() -> None:
    """Write generated/reference videos, classifier traces and a run manifest."""
    repo = Path(__file__).resolve().parents[2]
    workspace = repo.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=workspace / "data/pick_trocar_teleop_success_validation")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1, help="Shared-WM batch; use consecutive recorded episodes")
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--steps", type=int, default=15, help="WM denoising steps, not action chunk length")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", default="0", help="Physical CUDA device index or UUID; single GPU only")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing results are never overwritten"
    )
    parser.add_argument("--dreamdojo-root", type=Path, default=workspace / "DreamDojo")
    parser.add_argument(
        "--wm-checkpoint",
        type=Path,
        default=workspace / "models/dreamdojo/lora_r32_scratch_lr3e-4_18k/checkpoints/iter_000018000/model_ema_bf16.pt",
    )
    parser.add_argument("--experiment", default="dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora")
    parser.add_argument("--reward-checkpoint", type=Path, default=workspace / "DreamDojo/outputs/milestone/v2/best.pt")
    parser.add_argument("--lam-checkpoint", type=Path, default=os.environ.get("DREAMDOJO_LAM_CHECKPOINT"))
    parser.add_argument(
        "--check-reset", action="store_true", help="Repeat first step with the same reset seed and compare exactly"
    )
    args = parser.parse_args()
    if args.lam_checkpoint is None:
        parser.error("Pass --lam-checkpoint or set DREAMDOJO_LAM_CHECKPOINT to the existing official LAM checkpoint")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if not args.gpu or "," in args.gpu:
        parser.error("--gpu must select exactly one CUDA device")
    if torch.cuda.is_initialized():
        raise RuntimeError("Select --gpu before CUDA initialization; run the demo in a fresh process")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    output = args.output.absolute()
    cfg = DreamDojoInferenceCfg(
        dreamdojo_root=str(args.dreamdojo_root.resolve()),
        dataset=str(args.dataset.resolve()),
        wm_checkpoint=str(args.wm_checkpoint.resolve()),
        reward_checkpoint=str(args.reward_checkpoint.resolve()),
        lam_checkpoint=str(args.lam_checkpoint.resolve()),
        experiment=args.experiment,
        max_chunks=args.chunks,
        num_inference_steps=args.steps,
        seed=args.seed,
    )
    cfg.validate()
    episodes = list(range(args.episode_index, args.episode_index + args.num_envs))
    recordings = [load_recording(args.dataset.resolve(), episode, args.chunks) for episode in episodes]
    recorded_actions = np.stack([recording[0] for recording in recordings], axis=1)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "config": asdict(cfg), "recordings": [r[2] for r in recordings], "steps": []}
    repositories = [("IsaacLab", repo), ("DreamDojo", args.dreamdojo_root)]
    manifest["git_commits"] = {
        name: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        for name, path in repositories
    }
    source_files = [
        Path(__file__).resolve(),
        *sorted((repo / "source/isaaclab_contrib/isaaclab_contrib/neural").glob("*.py")),
        *sorted((repo / "source/isaaclab_tasks/isaaclab_tasks/contrib/dreamdojo_trocar").glob("*.py")),
        repo / "source/isaaclab_tasks/isaaclab_tasks/contrib/dreamdojo_trocar/config/dreamdojo_wm.yaml",
    ]
    manifest["source_sha256"] = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
    }
    manifest["scope"] = (
        "native DirectRLEnv + SimulationContext + neural backend; recorded actions; no robot/policy/training"
    )
    manifest_path = output / "results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    print(f"Loading real WM in native environment process {os.getpid()}", flush=True)
    environment = None
    try:
        native_cfg = make_dreamdojo_env_cfg(cfg, num_envs=args.num_envs)
        native_cfg.reset_options = {"episode_indices": episodes}
        environment = gym.make("Isaac-DreamDojo-Trocar-Direct-v0", cfg=native_cfg, render_mode="rgb_array")
        native = environment.unwrapped
        assert isinstance(native, DirectRLEnv)
        assert isinstance(native.sim, SimulationContext)
        assert native.sim.physics_manager is DreamDojoPhysicsManager
        assert all(getattr(NeuralDirectEnv, name) is getattr(DirectRLEnv, name) for name in ("step", "close"))
        manifest["native_integration"] = {
            "environment": f"{type(native).__module__}.{type(native).__name__}",
            "simulation_context": f"{type(native.sim).__module__}.{type(native.sim).__name__}",
            "manager": f"{native.sim.physics_manager.__module__}.{native.sim.physics_manager.__name__}",
            "inherited_core_methods": ["step", "close"],
            "reset": "forward reset options then delegate to native DirectRLEnv.reset",
            "scene_registry_keys": list(native.scene.keys()),
            "scene_assets": [key for key in native.scene.keys() if native.scene[key] is not None],
            "num_envs": native.num_envs,
            "step_dt": native.step_dt,
            "three_dimensional_transforms": native.sim.get_scene_data_provider().transform_count,
        }
        manifest["startup_seconds"] = time.perf_counter() - started
        obs, reset_extras = environment.reset(seed=args.seed)
        reset_info = reset_extras["reset_info"]
        np.testing.assert_array_equal(reset_info["start_frame"], [r[2]["first_image_frame_30hz"] for r in recordings])
        generated = [obs["policy"]["main_images"].cpu().numpy().copy()]
        states = [obs["policy"]["states"].cpu().numpy().copy()]
        probabilities, command_rewards, actions = [], [], []
        first_step = None
        for index in range(args.chunks):
            action_tensor = torch.from_numpy(recorded_actions[index])
            action = action_tensor.numpy().copy()
            actions.append(action)
            obs, reward_tensor, terminated_tensor, truncated_tensor, extras = environment.step(action_tensor)
            info = extras["neural_transition"]
            terminal_obs = extras.get("final_obs", obs)
            terminal_arrays = {key: value.cpu().numpy() for key, value in terminal_obs["policy"].items()}
            assert info["frames"].shape == (args.num_envs, 6, 480, 640, 3)
            np.testing.assert_array_equal(terminal_arrays["states"], action[:, -1])
            np.testing.assert_array_equal(terminal_arrays["main_images"], info["frames"][:, -1])
            assert native.sim.get_physics_step_count() == index + 1
            assert native.common_step_counter == index + 1
            generated.extend(info["frames"].swapaxes(0, 1))
            states.append(terminal_arrays["states"].copy())
            probabilities.append(info["probabilities"])
            command_rewards.append(info["command_rewards"])
            if index == 0:
                first_step = (info["frames"].copy(), info["probabilities"].copy(), reward_tensor.numpy().copy())
            record = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in info.items()
                if key not in ("frames", "probabilities", "command_rewards", "main_images", "states")
            }
            record.update(
                chunk=index + 1,
                reward=reward_tensor.tolist(),
                terminated=terminated_tensor.tolist(),
                truncated=truncated_tensor.tolist(),
                native_episode_length=native.episode_length_buf.tolist(),
            )
            manifest["steps"].append(record)
            print(json.dumps(record), flush=True)
            if (terminated_tensor | truncated_tensor).any():
                break
        if not truncated_tensor.all() or terminated_tensor.any() or len(manifest["steps"]) != args.chunks:
            raise RuntimeError("Unexpected demo termination or time-limit behavior")
        assert (native.episode_length_buf == 0).all()  # Reset by the unmodified core step.
        np.testing.assert_array_equal(obs["policy"]["main_images"].cpu().numpy(), generated[0])
        np.testing.assert_array_equal(obs["policy"]["states"].cpu().numpy(), states[0])
        manifest["native_terminal_obs_check"] = "final_obs_keeps_last_frame; returned_obs_is_reset_image"
        # Serialize useful results before the optional deterministic reset check.
        for slot, (_, reference, _) in enumerate(recordings):
            target = output if args.num_envs == 1 else output / f"env_{slot:03d}"
            target.mkdir(exist_ok=True)
            with imageio.get_writer(target / "generated.mp4", fps=15) as writer:
                for frame in generated:
                    writer.append_data(frame[slot])
            with imageio.get_writer(target / "reference.mp4", fps=15) as writer:
                for frame in reference:
                    writer.append_data(frame)
            with imageio.get_writer(target / "comparison.mp4", fps=15) as writer:
                for real, predicted in zip(reference, generated, strict=True):
                    writer.append_data(comparison_frame(real, predicted[slot]))
            np.savez_compressed(
                target / "trace.npz",
                actions=np.stack(actions)[:, slot],
                states=np.stack(states)[:, slot],
                probabilities=np.stack(probabilities)[:, slot],
                command_rewards=np.stack(command_rewards)[:, slot],
            )
        if args.check_reset:
            first_action = torch.from_numpy(recorded_actions[0])
            # First test core automatic reset by stepping without a manual reset.
            _, repeated_reward, _, _, repeated = environment.step(first_action)
            np.testing.assert_array_equal(repeated["neural_transition"]["frames"], first_step[0])
            np.testing.assert_array_equal(repeated["neural_transition"]["probabilities"], first_step[1])
            np.testing.assert_array_equal(repeated_reward.numpy(), first_step[2])
            manifest["native_auto_reset_check"] = "exact_first_step_match_without_manual_reset"
            environment.reset(seed=args.seed)
            _, repeated_reward, _, _, repeated = environment.step(first_action)
            np.testing.assert_array_equal(repeated["neural_transition"]["frames"], first_step[0])
            np.testing.assert_array_equal(repeated["neural_transition"]["probabilities"], first_step[1])
            np.testing.assert_array_equal(repeated_reward.numpy(), first_step[2])
            manifest["deterministic_reset_check"] = "exact_match_frames_probabilities_reward"
        manifest["native_total_steps"] = native.common_step_counter
        manifest.update(
            status="passed",
            video_frames=len(generated),
            video_fps=15,
            rollout_seconds=args.chunks * 0.4,
            reward_total=np.asarray([s["reward"] for s in manifest["steps"]]).sum(axis=0).tolist(),
            classifier_predicted_success=manifest["steps"][-1]["is_success"],
        )
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            manifest["native_context_released"] = SimulationContext.instance() is None
            manifest["total_wall_seconds"] = time.perf_counter() - started
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS: {output}", flush=True)


if __name__ == "__main__":
    main()
