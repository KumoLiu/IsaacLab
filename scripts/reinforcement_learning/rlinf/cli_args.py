# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command line argument utilities for RLinf integration with IsaacLab."""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def add_rlinf_args(parser: argparse.ArgumentParser):
    """Add RLinf arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rlinf", description="Arguments for RLinf agent.")
    # -- config arguments
    arg_group.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the RLinf configuration directory (for Hydra).",
    )
    arg_group.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Name of the RLinf configuration file (without .yaml extension).",
    )
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--resume_dir", type=str, default=None, help="Directory to resume training from.")
    arg_group.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path for evaluation.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "swanlab"},
        help="Logger module to use.",
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb."
    )
    # -- training arguments
    arg_group.add_argument(
        "--only_eval", action="store_true", default=False, help="Only run evaluation without training."
    )
    # -- cluster arguments
    arg_group.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for distributed training.")


def update_rlinf_cfg(agent_cfg, args_cli: argparse.Namespace):
    """Update configuration for RLinf agent based on CLI inputs.

    Supports both OmegaConf DictConfig (for Hydra configs) and
    dataclass configs (like RLinfPPORunnerCfg).

    Args:
        agent_cfg: The configuration for RLinf agent (DictConfig or dataclass).
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RLinf agent based on inputs.
    """
    from dataclasses import is_dataclass

    # Check if it's a dataclass (IsaacLab style) or DictConfig (Hydra style)
    if is_dataclass(agent_cfg):
        return _update_dataclass_cfg(agent_cfg, args_cli)
    else:
        return _update_omegaconf_cfg(agent_cfg, args_cli)


def _update_dataclass_cfg(agent_cfg, args_cli: argparse.Namespace):
    """Update dataclass configuration (RLinfPPORunnerCfg style)."""
    # Override seed
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed

    # Override environment settings
    if hasattr(args_cli, "num_envs") and args_cli.num_envs is not None:
        agent_cfg.env.total_num_envs = args_cli.num_envs

    # Override runner settings
    if args_cli.resume:
        agent_cfg.runner.resume = True
    if args_cli.resume_dir is not None:
        agent_cfg.runner.resume_dir = args_cli.resume_dir
    if args_cli.ckpt_path is not None:
        agent_cfg.runner.ckpt_path = args_cli.ckpt_path
    if args_cli.only_eval:
        agent_cfg.runner.only_eval = True

    # Override experiment name
    if args_cli.experiment_name is not None:
        agent_cfg.logger.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        if hasattr(agent_cfg.logger, "run_name"):
            agent_cfg.logger.run_name = args_cli.run_name

    # Override logger settings
    if args_cli.logger is not None:
        agent_cfg.logger.logger_backends = [args_cli.logger]
    if args_cli.log_project_name is not None:
        agent_cfg.logger.project_name = args_cli.log_project_name

    # Override max iterations
    if hasattr(args_cli, "max_iterations") and args_cli.max_iterations is not None:
        agent_cfg.runner.max_epochs = args_cli.max_iterations

    # Override cluster settings
    if args_cli.num_nodes is not None:
        agent_cfg.cluster.num_nodes = args_cli.num_nodes

    return agent_cfg


def _update_omegaconf_cfg(agent_cfg, args_cli: argparse.Namespace):
    """Update OmegaConf DictConfig configuration (Hydra style)."""
    from omegaconf import open_dict

    with open_dict(agent_cfg):
        # Override seed
        if hasattr(args_cli, "seed") and args_cli.seed is not None:
            if args_cli.seed == -1:
                args_cli.seed = random.randint(0, 10000)
            agent_cfg.actor.seed = args_cli.seed

        # Override environment settings
        if hasattr(args_cli, "num_envs") and args_cli.num_envs is not None:
            agent_cfg.env.train.total_num_envs = args_cli.num_envs
            agent_cfg.env.eval.total_num_envs = args_cli.num_envs

        # Override runner settings
        if args_cli.resume:
            agent_cfg.runner.resume = True
        if args_cli.resume_dir is not None:
            agent_cfg.runner.resume_dir = args_cli.resume_dir
        if args_cli.ckpt_path is not None:
            agent_cfg.runner.ckpt_path = args_cli.ckpt_path
        if args_cli.only_eval:
            agent_cfg.runner.only_eval = True

        # Override experiment name
        if args_cli.experiment_name is not None:
            agent_cfg.runner.logger.experiment_name = args_cli.experiment_name
        if args_cli.run_name is not None:
            agent_cfg.runner.logger.run_name = args_cli.run_name

        # Override logger settings
        if args_cli.logger is not None:
            agent_cfg.runner.logger.logger_backends = [args_cli.logger]
        if args_cli.log_project_name is not None:
            agent_cfg.runner.logger.project_name = args_cli.log_project_name

        # Override max iterations
        if hasattr(args_cli, "max_iterations") and args_cli.max_iterations is not None:
            agent_cfg.runner.max_epochs = args_cli.max_iterations

        # Override cluster settings
        if args_cli.num_nodes is not None:
            agent_cfg.cluster.num_nodes = args_cli.num_nodes

    return agent_cfg


def get_hydra_overrides(args_cli: argparse.Namespace) -> list[str]:
    """Generate Hydra override strings from CLI arguments.

    Args:
        args_cli: The command line arguments.

    Returns:
        List of Hydra override strings.
    """
    overrides = []

    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        overrides.append(f"actor.seed={args_cli.seed}")

    if hasattr(args_cli, "num_envs") and args_cli.num_envs is not None:
        overrides.append(f"env.train.total_num_envs={args_cli.num_envs}")
        overrides.append(f"env.eval.total_num_envs={args_cli.num_envs}")

    if args_cli.resume_dir is not None:
        overrides.append(f"runner.resume_dir={args_cli.resume_dir}")

    if args_cli.ckpt_path is not None:
        overrides.append(f"runner.ckpt_path={args_cli.ckpt_path}")

    if args_cli.only_eval:
        overrides.append("runner.only_eval=True")

    if args_cli.experiment_name is not None:
        overrides.append(f"runner.logger.experiment_name={args_cli.experiment_name}")

    if hasattr(args_cli, "max_iterations") and args_cli.max_iterations is not None:
        overrides.append(f"runner.max_epochs={args_cli.max_iterations}")

    if args_cli.num_nodes is not None:
        overrides.append(f"cluster.num_nodes={args_cli.num_nodes}")

    return overrides
