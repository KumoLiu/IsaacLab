# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate/play a checkpoint of an RL agent from RLinf.

This script provides an interface to launch RLinf evaluation for IsaacLab environments
following the standard IsaacLab reinforcement learning script pattern.

Usage:
    # Using IsaacLab task directly
    python play.py --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
        --ckpt_path /path/to/checkpoint --num_envs 4

    # Using RLinf config file
    python play.py --config_name isaaclab_franka_stack_cube_ppo_gr00t_demo \
        --ckpt_path /path/to/checkpoint

For full RLinf evaluation with all features, use the RLinf launcher directly:
    cd examples/embodiment
    bash eval_embodiment.sh isaaclab_franka_stack_cube_ppo_gr00t_demo
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RLinf.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (IsaacLab environment ID).")
parser.add_argument(
    "--agent", type=str, default="rlinf_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RLinf cli arguments
cli_args.add_rlinf_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Force only_eval mode
args_cli.only_eval = True

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
from rl_cfg import RLinfPPORunnerCfg
from vecenv_wrapper import RLinfVecEnvWrapper

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RLinfPPORunnerCfg | dict):
    """Evaluate with RLinf agent."""
    # Convert agent_cfg to OmegaConf if it's a dataclass
    if hasattr(agent_cfg, "to_dict"):
        agent_cfg = OmegaConf.create(agent_cfg.to_dict())
    elif isinstance(agent_cfg, dict):
        agent_cfg = OmegaConf.create(agent_cfg)

    # Grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # Override configurations with CLI arguments
    agent_cfg = cli_args.update_rlinf_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Set the environment seed
    seed = agent_cfg.get("seed", 42)
    env_cfg.seed = seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Specify directory for logging experiments
    experiment_name = agent_cfg.logger.get("experiment_name", "rlinf_experiment")
    log_root_path = os.path.join("logs", "rlinf", experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # Get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        print("[INFO] Pre-trained checkpoints are not yet available for RLinf.")
        print("[INFO] Please specify a checkpoint path using --ckpt_path.")
        return
    elif args_cli.ckpt_path:
        resume_path = retrieve_file_path(args_cli.ckpt_path)
    elif args_cli.resume_dir:
        resume_path = args_cli.resume_dir
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            agent_cfg.runner.get("load_run", None),
            agent_cfg.runner.get("checkpoint", None),
        )

    if resume_path:
        log_dir = os.path.dirname(resume_path)
    else:
        log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_eval")

    # Set the log directory for the environment
    env_cfg.log_dir = log_dir

    # Create Isaac environment using gym.make
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Get environment step dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Get observation mapping from config
    obs_map_cfg = agent_cfg.get("obs_map", {})
    obs_keys_mapping = {}
    if obs_map_cfg.get("main_images"):
        obs_keys_mapping["main_images"] = obs_map_cfg.main_images
    if obs_map_cfg.get("wrist_images"):
        obs_keys_mapping["wrist_images"] = obs_map_cfg.wrist_images

    state_keys = obs_map_cfg.get("state_keys", [])
    task_description = agent_cfg.env.get("task_description", "")

    # Wrap environment for RLinf
    env = RLinfVecEnvWrapper(
        env,
        task_description=task_description,
        obs_keys_mapping=obs_keys_mapping,
        state_keys=state_keys,
        convert_quat_to_axisangle=obs_map_cfg.get("convert_quat_to_axisangle", True),
    )

    print(f"[INFO] Environment created with {env.num_envs} parallel environments.")
    print(f"[INFO] Action dimension: {env.action_dim}")
    print(f"[INFO] Device: {env.device}")

    # Check if we should use RLinf's distributed evaluation
    use_rlinf_distributed = (
        agent_cfg.get("cluster", {}).get("num_nodes", 1) > 1
        or len(agent_cfg.get("cluster", {}).get("component_placement", {})) > 1
    )

    if use_rlinf_distributed and resume_path:
        print(f"\n[INFO] Starting RLinf distributed evaluation...")
        print(f"[INFO] Loading checkpoint from: {resume_path}")

        # Import RLinf components
        from rlinf.config import validate_cfg
        from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
        from rlinf.scheduler import Cluster
        from rlinf.utils.placement import HybridComponentPlacement
        from rlinf.workers.env.env_worker import EnvWorker
        from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

        # Close the environment we created (RLinf will create its own)
        env.close()

        # Build full RLinf config for evaluation
        rlinf_cfg = OmegaConf.create({
            "cluster": OmegaConf.to_container(agent_cfg.cluster, resolve=True),
            "runner": {
                **OmegaConf.to_container(agent_cfg.runner, resolve=True),
                "only_eval": True,
                "ckpt_path": resume_path,
            },
            "algorithm": OmegaConf.to_container(agent_cfg.algorithm, resolve=True),
            "actor": {
                "model": OmegaConf.to_container(agent_cfg.model, resolve=True),
            },
            "rollout": {
                "group_name": "RolloutGroup",
                "backend": "huggingface",
                "model": {"model_path": resume_path},
            },
            "env": {
                "group_name": "EnvGroup",
                "train": {
                    "env_type": "isaaclab",
                    "total_num_envs": env_cfg.scene.num_envs,
                    "init_params": {"id": args_cli.task},
                },
                "eval": {
                    "env_type": "isaaclab",
                    "total_num_envs": env_cfg.scene.num_envs,
                    "init_params": {"id": args_cli.task},
                    "video_cfg": {"save_video": args_cli.video},
                },
            },
        })

        # Validate and run evaluation
        rlinf_cfg.runner.only_eval = True
        rlinf_cfg = validate_cfg(rlinf_cfg)

        cluster = Cluster(cluster_cfg=rlinf_cfg.cluster)
        component_placement = HybridComponentPlacement(rlinf_cfg, cluster)

        # Create worker groups for evaluation (no actor needed)
        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = MultiStepRolloutWorker.create_group(rlinf_cfg).launch(
            cluster, name=rlinf_cfg.rollout.group_name, placement_strategy=rollout_placement
        )

        env_placement = component_placement.get_strategy("env")
        env_group = EnvWorker.create_group(rlinf_cfg).launch(
            cluster, name=rlinf_cfg.env.group_name, placement_strategy=env_placement
        )

        # Run evaluation
        runner = EmbodiedEvalRunner(
            cfg=rlinf_cfg,
            rollout=rollout_group,
            env=env_group,
        )
        runner.init_workers()
        runner.run()

    else:
        # Simple evaluation loop (for testing and validation)
        print("\n[INFO] Running simple evaluation loop...")

        if resume_path:
            print(f"[INFO] Loading checkpoint from: {resume_path}")

            # Try to load the policy
            try:
                checkpoint = torch.load(resume_path, map_location=env.device)
                print(f"[INFO] Checkpoint keys: {list(checkpoint.keys())}")

                # For now, use random policy as placeholder
                print("[WARNING] Using random policy - actual policy loading requires model definition.")
                policy = lambda obs: torch.randn(env.num_envs, env.action_dim, device=env.device) * 0.1

            except Exception as e:
                print(f"[WARNING] Could not load checkpoint: {e}")
                print("[INFO] Using random policy for validation.")
                policy = lambda obs: torch.randn(env.num_envs, env.action_dim, device=env.device) * 0.1
        else:
            print("[INFO] No checkpoint specified. Using random policy for validation.")
            policy = lambda obs: torch.randn(env.num_envs, env.action_dim, device=env.device) * 0.1

        # Reset environment
        obs, _ = env.reset()
        timestep = 0
        total_reward = torch.zeros(env.num_envs, device=env.device)

        # Simulate environment
        while simulation_app.is_running():
            start_time = time.time()

            # Run everything in inference mode
            with torch.inference_mode():
                # Get actions from policy
                if isinstance(obs, dict) and "states" in obs:
                    actions = policy(obs["states"])
                else:
                    actions = policy(obs)

                # Environment stepping
                obs, rewards, terminated, truncated, info = env.step(actions)
                total_reward += rewards

            timestep += 1

            # Print progress
            if timestep % 50 == 0:
                print(f"  Step {timestep}: mean_reward={rewards.mean().item():.4f}, "
                      f"total_reward={total_reward.mean().item():.4f}")

            # Check for video recording completion
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # Check for episode completion (for non-video mode)
            if not args_cli.video and (terminated | truncated).all():
                print(f"\n[INFO] All episodes completed at step {timestep}")
                break

            # Time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        # Print final metrics
        if "episode" in info:
            print("\n[INFO] Episode Metrics:")
            for key, value in info["episode"].items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.float().mean().item():.4f}")

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
