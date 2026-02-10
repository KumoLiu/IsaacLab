# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RLinf.

This script launches RLinf distributed training for IsaacLab tasks.
Tasks can be either:
1. Registered in IsaacLab with `rlinf_cfg_entry_point` - will be auto-registered into RLinf
2. Already registered in RLinf's REGISTER_ISAACLAB_ENVS

Usage:
    # List available tasks
    python train.py --list_tasks

    # Train an IsaacLab task (auto-registers into RLinf)
    python train.py --task Isaac-MyTask-v0 --model_path /path/to/model

    # Train with custom config
    python train.py --task Isaac-MyTask-v0 --model_path /path/to/model \\
        --num_envs 64 --max_iterations 1000

Note:
    RLinf training requires a pretrained VLA model (e.g., GR00T, OpenVLA).
    The model_path should point to a HuggingFace format checkpoint directory.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RLinf.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment (overrides config if set)")
parser.add_argument("--max_epochs", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--list_tasks", action="store_true", default=False, help="List all available tasks and exit.")
parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model checkpoint (required).")

# append RLinf cli arguments
cli_args.add_rlinf_args(parser)
args_cli = parser.parse_args()

# Handle --list_tasks before any heavy imports
if args_cli.list_tasks:
    print("\n" + "=" * 60)
    print("Available RLinf Tasks")
    print("=" * 60)

    # List RLinf registered tasks
    print("\n[RLinf Registered Tasks]")
    try:
        from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

        for task_id in sorted(REGISTER_ISAACLAB_ENVS.keys()):
            print(f"  - {task_id}")
    except ImportError:
        print("  (Could not import RLinf registry)")

    print("\n" + "=" * 60)
    sys.exit(0)

# task is optional if RLINF_CONFIG_FILE is set (task_id comes from YAML)
# model_path is also optional if set in YAML

"""Rest of the script - launch RLinf training."""

import logging
import torch.multiprocessing as mp

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)


def get_config_path_and_name(args_cli) -> tuple[Path, str]:
    """Get config path and name.
    
    Priority:
    1. RLINF_CONFIG_FILE environment variable (full path, set by run.sh)
    2. CLI --config_name argument (looks in rlinf/config directory)
    """
    config_file = os.environ.get("RLINF_CONFIG_FILE", "")
    if config_file:
        return Path(config_file).parent, Path(config_file).stem
    
    script_dir = Path(__file__).parent.absolute()
    config_path = script_dir / "config"

    if args_cli.config_name:
        return config_path, args_cli.config_name

    raise FileNotFoundError("No config found. Set RLINF_CONFIG_FILE or --config_name")


def main():
    """Launch RLinf training."""
    # Get config (task_id is read from YAML)
    config_path, config_name = get_config_path_and_name(args_cli)
    print(f"[INFO] Using config: {config_name}")
    print(f"[INFO] Config path: {config_path}")

    # Initialize Hydra and load config
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(config_path), version_base="1.1")
    cfg = compose(config_name=config_name)

    # Get task_id from config
    task_id = cfg.env.train.init_params.id
    print(f"[INFO] Task: {task_id}")

    # Setup logging directory (use RLINF_LOG_DIR from run.sh if available)
    if os.environ.get("RLINF_LOG_DIR"):
        log_dir = Path(os.environ["RLINF_LOG_DIR"])
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        log_dir = Path("logs") / "rlinf" / f"{timestamp}-{task_id.replace('/', '_')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Apply runtime overrides from CLI arguments
    with open_dict(cfg):
        cfg.runner.logger.log_path = str(log_dir)
        
        # Override from CLI if provided
        if args_cli.num_envs is not None:
            cfg.env.train.total_num_envs = args_cli.num_envs
            cfg.env.eval.total_num_envs = args_cli.num_envs
        if args_cli.seed is not None:
            cfg.actor.seed = args_cli.seed
        if args_cli.max_epochs is not None:
            cfg.runner.max_epochs = args_cli.max_epochs
        if args_cli.model_path is not None:
            cfg.actor.model.model_path = args_cli.model_path
            cfg.rollout.model.model_path = args_cli.model_path
        if args_cli.only_eval:
            cfg.runner.only_eval = True
        if args_cli.resume_dir:
            cfg.runner.resume_dir = args_cli.resume_dir

    # Validate config
    cfg = validate_cfg(cfg)

    # Print config summary
    print("\n" + "=" * 60)
    print("RLinf Training Configuration")
    print("=" * 60)
    print(f"  Task: {cfg.env.train.init_params.id}")
    print(f"  Num envs: {cfg.env.train.total_num_envs}")
    print(f"  Max epochs: {cfg.runner.max_epochs}")
    print(f"  Model: {cfg.actor.model.model_path}")
    print(f"  Algorithm: {cfg.algorithm.loss_type}")
    print("=" * 60 + "\n")

    # Create cluster and component placement
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker
    actor_placement = component_placement.get_strategy("actor")
    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    else:
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

        actor_worker_cls = EmbodiedFSDPActor

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

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

    # Create and run training
    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
