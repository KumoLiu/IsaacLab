# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded actions through Cosmos3 to generate new video observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from isaaclab_contrib.neural import NeuralDirectEnv
from isaaclab_contrib.neural.cosmos3 import Cosmos3InferenceCfg, make_cosmos3_env_cfg


def main() -> None:
    """Write one generated video and resolved run metadata; no policy or RL library is required."""
    parser = argparse.ArgumentParser(description=__doc__)
    sample = (
        Path(__file__).resolve().parents[2]
        / "source/isaaclab_tasks/isaaclab_tasks/contrib/cosmos3_medical_pad_unfold/assets"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local DCP root or complete HF export")
    parser.add_argument("--config-file", type=Path, help="Saved model YAML; required for a DCP checkpoint")
    parser.add_argument("--initial-image", type=Path, default=sample / "validation_initial.png")
    parser.add_argument("--prompt", default="unfold the medical nursing pad")
    parser.add_argument(
        "--actions", type=Path, default=sample / "replay_actions.json", help="normalized_unified_v1 JSON sidecar"
    )
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--mode", choices=("window", "session"), default="window")
    parser.add_argument(
        "--chunk-size", type=int, default=16, help="Action rows per step; use the session checkpoint's chunk"
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=(480, 640), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--resolution", default="480", help="Native Cosmos resolution tier")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--domain", default="hand_pose")
    parser.add_argument(
        "--steps", type=int, help="Window-mode denoising steps (default 30); session uses its saved sampler"
    )
    parser.add_argument("--shift", type=float, help="Window sampler shift (default 10); omit for session mode")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--regular-weights", action="store_true", help="Load regular DCP weights instead of EMA")
    args = parser.parse_args()
    payload = json.loads(args.actions.read_text())
    if not isinstance(payload, dict) or payload.get("action_space") != "normalized_unified_v1":
        parser.error("Actions must be a normalized_unified_v1 sidecar, not raw robot joints")
    actions = np.asarray(payload["action"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 59 or not np.isfinite(actions).all():
        parser.error("Actions must be finite with shape (T,59)")
    if args.chunk_size < 1 or not len(actions) or len(actions) % args.chunk_size:
        parser.error("Action rows must be a positive exact multiple of --chunk-size")
    domains = payload.get("domain_names", args.domain)
    if isinstance(domains, str):
        domains = [domains] * len(actions)
    if not isinstance(domains, list) or len(domains) not in (1, len(actions)) or any(d != args.domain for d in domains):
        parser.error("This demo uses one fixed --domain; sidecar domains must agree")
    inference = Cosmos3InferenceCfg(
        checkpoint_path=str(args.checkpoint.resolve()),
        config_file=str(args.config_file.resolve()) if args.config_file else None,
        output_dir=str(args.output.resolve()),
        mode=args.mode,
        chunk_size=args.chunk_size,
        image_size=tuple(args.image_size),
        fps=args.fps,
        domain_name=args.domain,
        resolution=args.resolution,
        num_inference_steps=args.steps,
        shift=args.shift,
        use_ema_weights=not args.regular_weights,
    )
    cfg = make_cosmos3_env_cfg(
        inference,
        initial_image=str(args.initial_image.resolve()),
        prompt=args.prompt,
        max_chunks=len(actions) // args.chunk_size,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    import imageio.v2 as imageio

    metadata = {"checkpoint": inference.checkpoint_path, "mode": args.mode, "steps": [], "task_reward": "none"}
    with (
        NeuralDirectEnv(cfg) as env,
        imageio.get_writer(args.output / "generated.mp4", fps=args.fps, macro_block_size=1) as writer,
    ):
        observation, _ = env.reset(seed=args.seed)
        writer.append_data(observation["policy"]["main_images"][0].numpy())
        for chunk in actions.reshape(-1, 1, args.chunk_size, 59):
            _, _, terminated, truncated, extras = env.step(torch.from_numpy(chunk))
            transition = extras["neural_transition"]
            for frame in transition["frames"][0]:
                writer.append_data(frame)
            metadata["steps"].append(
                {
                    "elapsed_seconds": float(transition["elapsed_seconds"][0]),
                    "terminated": bool(terminated[0]),
                    "truncated": bool(truncated[0]),
                }
            )
    (args.output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
