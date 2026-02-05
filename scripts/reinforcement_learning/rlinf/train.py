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
import json
import sys
from datetime import datetime
from pathlib import Path

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RLinf.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rlinf_cfg_entry_point", help="Name of the RLinf agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=1234, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--list_tasks", action="store_true", default=False, help="List all available tasks and exit.")
parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model checkpoint (required).")
# parser.add_argument("--config_name", type=str, default=None, help="RLinf Hydra config name (optional).")
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

    # List IsaacLab registered tasks with rlinf_cfg_entry_point
    print("\n[IsaacLab Tasks with rlinf_cfg_entry_point]")
    print("  (Requires IsaacLab simulator - cannot list without launching)")
    print("  Known tasks:")
    print("  - Isaac-Install-Trocar-G129-Dex3-RLinf-v0")
    print("  - Isaac-Install-Trocar-G129-Dex3-RLinf-Eval-v0")
    print("")
    print("  To add your task, register with 'rlinf_cfg_entry_point' in gym.register()")

    print("\n" + "=" * 60)
    sys.exit(0)

# Validate arguments
if args_cli.task is None:
    print("[ERROR] Please specify a task with --task")
    print("[INFO] Use --list_tasks to see available tasks")
    sys.exit(1)

if args_cli.model_path is None:
    print("[ERROR] Please specify a model checkpoint with --model_path")
    print("[INFO] RLinf training requires a pretrained VLA model (e.g., GR00T)")
    sys.exit(1)

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


def get_task_source(task_name: str) -> tuple[str, bool]:
    """Determine task source and availability.
    
    Returns:
        Tuple of (source_description, is_available)
    """
    import os
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    # Check if already in RLinf registry
    if task_name in REGISTER_ISAACLAB_ENVS:
        return "rlinf", True

    # Check if task has rlinf_cfg_entry_point (will be auto-registered via extension)
    try:
        import gymnasium as gym
        if task_name in gym.registry:
            task_spec = gym.registry[task_name]
            if task_spec.kwargs and "rlinf_cfg_entry_point" in task_spec.kwargs:
                return "isaaclab (auto-register via RLINF_EXT_MODULE)", True
    except Exception:
        pass

    # If RLINF_EXT_MODULE is set and task looks like an IsaacLab task,
    # trust that it will be auto-registered when Workers start
    # (isaaclab_tasks can't be fully imported without Omniverse running)
    if os.environ.get("RLINF_EXT_MODULE") and task_name.startswith("Isaac-"):
        return "isaaclab (will be auto-registered by extension)", True

    return "unknown", False


def find_or_create_config(task_id: str, args_cli) -> tuple[Path, str]:
    """Find existing config or create a new one for the task.

    Returns:
        Tuple of (config_path, config_name)
    """
    # Look for existing config in examples/embodiment/config
    script_dir = Path(__file__).parent.absolute()
    repo_root = script_dir.parent.parent.parent.parent
    config_path = repo_root / "examples" / "embodiment" / "config"

    if args_cli.config_name:
        return config_path, args_cli.config_name

    # Search for matching config
    for config_file in config_path.glob("isaaclab*.yaml"):
        with open(config_file) as f:
            content = f.read()
            if task_id in content:
                return config_path, config_file.stem

    # Use default config
    default_config = "isaaclab_ppo_gr00t_install_trocar"
    if (config_path / f"{default_config}.yaml").exists():
        logger.info(f"No specific config for '{task_id}', using '{default_config}'")
        return config_path, default_config

    raise FileNotFoundError(f"No RLinf config found for task '{task_id}' and no default available")


def main():
    """Launch RLinf training."""
    task_id = args_cli.task
    task_source, is_available = get_task_source(task_id)
    print(f"[INFO] Task '{task_id}' source: {task_source}")

    if not is_available:
        print(f"[ERROR] Task '{task_id}' not found")
        print("  Options:")
        print("  1. Use a task registered in RLinf's REGISTER_ISAACLAB_ENVS")
        print("  2. Register your IsaacLab task with rlinf_cfg_entry_point")
        print("     and ensure RLINF_EXT_MODULE=isaaclab_rl.rlinf.extension is set")
        print("")
        print("  Available RLinf tasks:")
        from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS
        for t in sorted(REGISTER_ISAACLAB_ENVS.keys()):
            print(f"    - {t}")
        sys.exit(1)

    # Find or create config
    config_path, config_name = find_or_create_config(task_id, args_cli)
    print(f"[INFO] Using config: {config_name}")
    print(f"[INFO] Config path: {config_path}")

    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    log_dir = Path("logs") / "rlinf" / f"{timestamp}-{task_id.replace('/', '_')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Initialize Hydra and load config
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(config_path), version_base="1.1")
    cfg = compose(config_name=config_name)

    # Apply CLI overrides
    with open_dict(cfg):
        # Set task ID
        cfg.env.train.init_params.id = task_id
        cfg.env.eval.init_params.id = task_id.replace("-v0", "-Eval-v0") if "-v0" in task_id else f"{task_id}-Eval"

        # Set model path
        cfg.actor.model.model_path = args_cli.model_path
        cfg.rollout.model.model_path = args_cli.model_path

        # Set logging
        cfg.runner.logger.log_path = str(log_dir)

        # Apply other CLI args
        if args_cli.num_envs is not None:
            cfg.env.train.total_num_envs = args_cli.num_envs
            cfg.env.eval.total_num_envs = args_cli.num_envs

        if args_cli.max_iterations is not None:
            cfg.runner.max_epochs = args_cli.max_iterations

        cfg.actor.seed = args_cli.seed

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

    # Save config
    OmegaConf.save(cfg, log_dir / "config.yaml")
    with open(log_dir / "config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

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

    # Create demo buffer if configured
    demo_buffer = None
    if cfg.get("data", None):
        from rlinf.data.datasets import create_rl_dataset

        demo_buffer, _ = create_rl_dataset(cfg, tokenizer=None)

    # Create and run training
    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        demo_buffer=demo_buffer,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
