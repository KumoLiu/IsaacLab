# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf integration scripts for IsaacLab.

This module provides scripts to train and evaluate RLinf agents with IsaacLab environments.

Scripts:
    train.py: Train an RL agent using RLinf's distributed architecture.
    play.py: Evaluate a trained RLinf agent checkpoint.

Key Components:
    RLinfVecEnvWrapper: Wrapper to convert IsaacLab environments to RLinf format.
    RLinfPPORunnerCfg: Configuration class for RLinf PPO training.

Example task registration with rlinf_cfg_entry_point:

.. code-block:: python

    gym.register(
        id="Isaac-MyTask-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.my_task_cfg:MyTaskEnvCfg",
            "rlinf_cfg_entry_point": f"{agents.__name__}.rlinf_ppo_cfg:RLinfPPORunnerCfg",
        },
    )
"""
