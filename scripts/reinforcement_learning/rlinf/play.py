# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained RLinf agent.

This script runs evaluation using RLinf's distributed infrastructure,
which is required for VLA model inference.

Usage:
    # Evaluate a trained checkpoint
    python play.py --task Isaac-MyTask-v0 --checkpoint /path/to/checkpoint

    # Evaluate with specific number of environments
    python play.py --task Isaac-MyTask-v0 --checkpoint /path/to/checkpoint --num_envs 8

Note:
    Evaluation requires the full RLinf infrastructure since VLA models
    are too large to run on a single GPU without FSDP.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained RLinf agent.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rlinf_cfg_entry_point", help="Name of the RLinf agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=1234, help="Seed used for the environment")
parser.add_argument(
    "--checkpoint", type=str, default=None, required=True, help="Path to the model checkpoint (required)."
)
parser.add_argument("--config_name", type=str, default=None, help="RLinf Hydra config name (optional).")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
# append RLinf cli arguments
cli_args.add_rlinf_args(parser)
args_cli = parser.parse_args()

# Validate arguments
if args_cli.task is None:
    print("[ERROR] Please specify a task with --task")
    sys.exit(1)

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


def get_task_source(task_name: str) -> str:
    """Determine if task is from IsaacLab or RLinf registry."""
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    if task_name in REGISTER_ISAACLAB_ENVS:
        return "rlinf"
    return "isaaclab"


def register_isaaclab_task(task_id: str, agent_entry_point: str) -> None:
    """Register an IsaacLab task into RLinf's registries."""
    from isaaclab_rl.rlinf import RLinfPPORunnerCfg, register_task_for_rlinf

    try:
        import gymnasium as gym

        if task_id not in gym.registry:
            logger.warning(f"Task '{task_id}' not found in gymnasium registry")
            return

        task_spec = gym.registry[task_id]
        if not task_spec.kwargs or agent_entry_point not in task_spec.kwargs:
            logger.warning(f"Task '{task_id}' has no '{agent_entry_point}'")
            return

        entry_point = task_spec.kwargs[agent_entry_point]
        if isinstance(entry_point, str):
            module_name, class_name = entry_point.rsplit(":", 1)
            import importlib

            module = importlib.import_module(module_name)
            agent_cfg = getattr(module, class_name)()
        else:
            agent_cfg = entry_point()

        if not isinstance(agent_cfg, RLinfPPORunnerCfg):
            agent_cfg = RLinfPPORunnerCfg()

        register_task_for_rlinf(task_id, agent_cfg)

    except Exception as e:
        logger.warning(f"Could not register task '{task_id}': {e}")


def find_or_create_config(task_id: str, args_cli) -> tuple[Path, str]:
    """Find existing config or use default."""
    script_dir = Path(__file__).parent.absolute()
    repo_root = script_dir.parent.parent.parent.parent
    config_path = repo_root / "examples" / "embodiment" / "config"

    if args_cli.config_name:
        return config_path, args_cli.config_name

    for config_file in config_path.glob("isaaclab*.yaml"):
        with open(config_file) as f:
            content = f.read()
            if task_id in content:
                return config_path, config_file.stem

    default_config = "isaaclab_ppo_gr00t_install_trocar"
    if (config_path / f"{default_config}.yaml").exists():
        return config_path, default_config

    raise FileNotFoundError(f"No RLinf config found for task '{task_id}'")


def main():
    """Launch RLinf evaluation."""
    task_id = args_cli.task
    task_source = get_task_source(task_id)
    print(f"[INFO] Task '{task_id}' source: {task_source}")

    # Register IsaacLab task if needed
    if task_source == "isaaclab":
        print(f"[INFO] Registering IsaacLab task '{task_id}' into RLinf...")
        register_isaaclab_task(task_id, args_cli.agent)

    # Find config
    config_path, config_name = find_or_create_config(task_id, args_cli)
    print(f"[INFO] Using config: {config_name}")

    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    log_dir = Path("logs") / "rlinf" / "eval" / f"{timestamp}-{task_id.replace('/', '_')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Initialize Hydra and load config
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(config_path), version_base="1.1")
    cfg = compose(config_name=config_name)

    # Apply CLI overrides for evaluation
    with open_dict(cfg):
        # Set evaluation mode
        cfg.runner.only_eval = True

        # Set task ID (use eval variant if exists)
        eval_task_id = task_id.replace("-v0", "-Eval-v0") if "-v0" in task_id else f"{task_id}-Eval"
        cfg.env.eval.init_params.id = eval_task_id

        # Set model path
        cfg.rollout.model.model_path = args_cli.checkpoint

        # Set logging
        cfg.runner.logger.log_path = str(log_dir)

        # Enable video saving
        cfg.env.eval.video_cfg.save_video = True
        cfg.env.eval.video_cfg.video_base_dir = str(log_dir / "videos")

        # Apply CLI args
        if args_cli.num_envs is not None:
            cfg.env.eval.total_num_envs = args_cli.num_envs

        cfg.actor.seed = args_cli.seed

    cfg = validate_cfg(cfg)

    # Print config summary
    print("\n" + "=" * 60)
    print("RLinf Evaluation Configuration")
    print("=" * 60)
    print(f"  Task: {cfg.env.eval.init_params.id}")
    print(f"  Num envs: {cfg.env.eval.total_num_envs}")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print(f"  Videos: {cfg.env.eval.video_cfg.video_base_dir}")
    print("=" * 60 + "\n")

    # Save config
    OmegaConf.save(cfg, log_dir / "config.yaml")
    with open(log_dir / "config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

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
