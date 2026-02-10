# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained RLinf agent.

This script runs evaluation using RLinf's distributed infrastructure,
which is required for VLA model inference.

Usage:
    # Evaluate a trained checkpoint
    bash play.sh --task Isaac-MyTask-v0 --checkpoint /path/to/checkpoint

    # Evaluate with specific number of environments
    bash play.sh --task Isaac-MyTask-v0 --checkpoint /path/to/checkpoint --num_envs 8

Note:
    Evaluation requires the full RLinf infrastructure since VLA models
    are too large to run on a single GPU without FSDP.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained RLinf agent.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (optional, read from config).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment (overrides config if set)")
parser.add_argument(
    "--model_path", type=str, default=None, help="Path to the model checkpoint (optional, can be set in config)."
)
parser.add_argument("--num_episodes", type=int, default=None, help="Number of evaluation episodes (overrides config if set).")
parser.add_argument("--video", action="store_true", default=False, help="Enable video recording.")
# append RLinf cli arguments
cli_args.add_rlinf_args(parser)
args_cli = parser.parse_args()


"""Rest of the script - launch RLinf evaluation."""

import logging
import torch.multiprocessing as mp

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)


def get_config_path_and_name(args_cli) -> tuple[Path, str]:
    """Get config path and name.
    
    Priority:
    1. RLINF_CONFIG_FILE environment variable (full path, set by play.sh)
    2. CLI --config_name argument (looks in rlinf/config directory)
    """
    config_file = os.environ.get("RLINF_CONFIG_FILE", "")
    if config_file:
        return Path(config_file).parent, Path(config_file).stem
    
    script_dir = Path(__file__).parent.absolute()
    config_path = script_dir / "config"

    if hasattr(args_cli, 'config_name') and args_cli.config_name:
        return config_path, args_cli.config_name

    raise FileNotFoundError("No config found. Set RLINF_CONFIG_FILE or --config_name")


def main():
    """Launch RLinf evaluation."""
    # Get config (task_id is read from YAML)
    config_path, config_name = get_config_path_and_name(args_cli)
    print(f"[INFO] Using config: {config_name}")
    print(f"[INFO] Config path: {config_path}")

    # Initialize Hydra and load config
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(config_path), version_base="1.1")
    cfg = compose(config_name=config_name)

    # Get task_id from config (eval task)
    task_id = cfg.env.eval.init_params.id
    print(f"[INFO] Task: {task_id}")

    # Setup logging directory (use RLINF_LOG_DIR from play.sh if available)
    if os.environ.get("RLINF_LOG_DIR"):
        log_dir = Path(os.environ["RLINF_LOG_DIR"])
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        log_dir = Path("logs") / "rlinf" / "eval" / f"{timestamp}-{task_id.replace('/', '_')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Apply runtime overrides
    with open_dict(cfg):
        # Set evaluation mode
        cfg.runner.only_eval = True
        
        # Set logging
        cfg.runner.logger.log_path = str(log_dir)

        # Override checkpoint if provided via CLI
        if args_cli.model_path:
            cfg.rollout.model.model_path = args_cli.model_path

        # Enable video saving if requested
        if args_cli.video:
            cfg.env.eval.video_cfg.save_video = True
            cfg.env.eval.video_cfg.video_base_dir = str(log_dir / "videos")

        # Apply CLI args
        if args_cli.num_envs is not None:
            cfg.env.eval.total_num_envs = args_cli.num_envs
        if args_cli.seed is not None:
            cfg.actor.seed = args_cli.seed
        if args_cli.num_episodes is not None:
            cfg.algorithm.eval_rollout_epoch = args_cli.num_episodes

    # Validate config
    cfg = validate_cfg(cfg)

    # Print config summary
    print("\n" + "=" * 60)
    print("RLinf Evaluation Configuration")
    print("=" * 60)
    print(f"  Task: {cfg.env.eval.init_params.id}")
    print(f"  Num envs: {cfg.env.eval.total_num_envs}")
    print(f"  Model: {cfg.rollout.model.model_path}")
    print(f"  Videos: {cfg.env.eval.video_cfg.save_video}")
    if cfg.env.eval.video_cfg.save_video:
        print(f"  Video dir: {cfg.env.eval.video_cfg.video_base_dir}")
    print("=" * 60 + "\n")

    # Create cluster and workers
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create rollout worker
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    # Run evaluation
    runner = EmbodiedEvalRunner(
        cfg=cfg,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()