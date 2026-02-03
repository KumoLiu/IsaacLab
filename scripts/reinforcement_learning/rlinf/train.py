# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic RLinf training script for IsaacLab tasks.

This script launches RLinf distributed training for any IsaacLab task that has
been registered in the RLinf environment registry.

Usage:
    # Train with a specific config
    python train.py --config_name isaaclab_ppo_gr00t_install_trocar
    
    # Train with overrides
    python train.py --config_name isaaclab_ppo_gr00t_install_trocar \
        --num_envs 56 --max_iterations 1000
    
    # Evaluation only
    python train.py --config_name isaaclab_ppo_gr00t_install_trocar \
        --only_eval --eval_policy_path /path/to/checkpoint

Similar to RSL-RL integration at:
    IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch.multiprocessing as mp

# Set multiprocessing start method before any other imports
mp.set_start_method("spawn", force=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RLinf agent on IsaacLab tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train.py --config_name isaaclab_ppo_gr00t_install_trocar
    
    # With custom parameters
    python train.py --config_name isaaclab_ppo_gr00t_install_trocar \\
        --num_envs 56 --max_iterations 1000
        
    # Evaluation mode
    python train.py --config_name isaaclab_ppo_gr00t_install_trocar \\
        --only_eval --eval_policy_path /path/to/checkpoint
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Name of the Hydra config file (without .yaml extension).",
    )
    
    # Optional overrides
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config directory. Defaults to examples/embodiment/config/.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of environments to simulate. Overrides config value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed used for training.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Maximum training iterations (epochs). Overrides config value.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint. Overrides config value.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save logs. Defaults to logs/<timestamp>-<config_name>.",
    )
    parser.add_argument(
        "--only_eval",
        action="store_true",
        default=False,
        help="Run evaluation only (no training).",
    )
    parser.add_argument(
        "--eval_policy_path",
        type=str,
        default=None,
        help="Path to policy checkpoint for evaluation.",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Directory to resume training from.",
    )
    
    # Hydra overrides (pass-through)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra config overrides in key=value format.",
    )
    
    return parser.parse_args()


def find_config_path(args) -> Path:
    """Find the config directory."""
    if args.config_path:
        config_path = Path(args.config_path)
        if config_path.exists():
            return config_path.absolute()
        raise FileNotFoundError(f"Config path not found: {config_path}")
    
    # Try different locations
    script_dir = Path(__file__).parent.absolute()
    isaaclab_dir = script_dir.parent.parent.parent
    repo_root = isaaclab_dir.parent
    
    candidates = [
        repo_root / "examples" / "embodiment" / "config",
        Path("examples/embodiment/config"),
        Path.cwd() / "examples" / "embodiment" / "config",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate.absolute()
    
    raise FileNotFoundError(
        f"Config directory not found. Tried: {candidates}\n"
        "Please specify --config_path explicitly."
    )


def main():
    """Main training function for RLinf on IsaacLab tasks."""
    args = parse_args()
    
    # Find config directory
    try:
        config_path = find_config_path(args)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    print(f"[INFO] Config directory: {config_path}")
    
    # Check if config file exists
    config_file = config_path / f"{args.config_name}.yaml"
    if not config_file.exists():
        print(f"[ERROR] Config file not found: {config_file}")
        available_configs = sorted([c.stem for c in config_path.glob("*.yaml")])
        print(f"[INFO] Available configs:")
        for cfg_name in available_configs:
            print(f"  - {cfg_name}")
        sys.exit(1)
    
    print(f"[INFO] Config file: {args.config_name}.yaml")
    
    # Setup log directory
    script_dir = Path(__file__).parent.absolute()
    isaaclab_dir = script_dir.parent.parent.parent
    repo_root = isaaclab_dir.parent
    
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        log_dir = repo_root / "logs" / f"{timestamp}-{args.config_name}"
    
    log_dir = log_dir.absolute()
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Log directory: {log_dir}")
    
    # Import RLinf components
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf, open_dict
        
        from rlinf.config import validate_cfg
        from rlinf.runners.embodied_runner import EmbodiedRunner
        from rlinf.scheduler import Cluster
        from rlinf.utils.placement import HybridComponentPlacement
        from rlinf.workers.env.env_worker import EnvWorker
        from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    except ImportError as e:
        print(f"[ERROR] Failed to import RLinf components: {e}")
        print("[ERROR] Make sure RLinf is in your PYTHONPATH.")
        print("[INFO] You can run: export PYTHONPATH=/path/to/RLinf-Orca:$PYTHONPATH")
        sys.exit(1)
    
    # Initialize Hydra and compose config
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(config_path), version_base="1.1")
    
    # Build overrides list
    override_list = list(args.overrides) if args.overrides else []
    override_list.append(f"runner.logger.log_path={log_dir}")
    
    cfg = compose(config_name=args.config_name, overrides=override_list)
    
    # Apply CLI overrides
    with open_dict(cfg):
        if args.num_envs is not None:
            cfg.env.train.total_num_envs = args.num_envs
            cfg.env.eval.total_num_envs = args.num_envs
        
        if args.max_iterations is not None:
            cfg.runner.max_epochs = args.max_iterations
        
        if args.model_path is not None:
            cfg.actor.model.model_path = args.model_path
            cfg.rollout.model.model_path = args.model_path
        
        cfg.actor.seed = args.seed
        
        if args.only_eval:
            cfg.runner.only_eval = True
        
        if args.eval_policy_path is not None:
            cfg.runner.eval_policy_path = args.eval_policy_path
        
        if args.resume_dir is not None:
            cfg.runner.resume_dir = args.resume_dir
    
    # Validate config
    cfg = validate_cfg(cfg)
    
    # Print config summary
    print("\n" + "=" * 60)
    print("RLinf Training Configuration")
    print("=" * 60)
    print(f"  Task: {cfg.env.train.init_params.id}")
    print(f"  Num envs (train): {cfg.env.train.total_num_envs}")
    print(f"  Num envs (eval): {cfg.env.eval.total_num_envs}")
    print(f"  Max epochs: {cfg.runner.max_epochs}")
    print(f"  Model path: {cfg.actor.model.model_path}")
    print(f"  Algorithm: {cfg.algorithm.loss_type}")
    print(f"  Only eval: {cfg.runner.only_eval}")
    print(f"  Log path: {cfg.runner.logger.log_path}")
    print("=" * 60 + "\n")
    
    # Save config
    OmegaConf.save(cfg, log_dir / "config.yaml")
    with open(log_dir / "config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
    print(f"[INFO] Config saved to: {log_dir}")
    
    # Create cluster and workers
    print("[INFO] Creating cluster...")
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)
    
    # Create actor worker group
    print("[INFO] Creating actor worker group...")
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
    
    # Create rollout worker group
    print("[INFO] Creating rollout worker group...")
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    
    # Create env worker group
    print("[INFO] Creating env worker group...")
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    
    # Create demo buffer if data config exists
    demo_buffer = None
    if cfg.get("data", None):
        print("[INFO] Creating demo buffer...")
        from rlinf.data.datasets import create_rl_dataset
        demo_buffer, _ = create_rl_dataset(cfg, tokenizer=None)
    
    # Create and run the training runner
    print("\n" + "=" * 60)
    print("Starting RLinf Training")
    print("=" * 60 + "\n")
    
    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        demo_buffer=demo_buffer,
    )
    
    runner.init_workers()
    runner.run()
    
    print("\n[INFO] Training completed!")
    print(f"[INFO] Logs saved to: {log_dir}")


if __name__ == "__main__":
    main()
