# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Install Trocar task with G129 + Dex3 robot.

This module registers the Install Trocar task in IsaacLab's gymnasium registry,
allowing it to be discovered and used through IsaacLab's standard task interfaces.

The environment configuration is imported from RLinf's task definitions to avoid
code duplication while enabling IsaacLab-native task discovery.
"""

import gymnasium as gym

from . import agents

# Import the environment configs from RLinf
# This allows us to reuse the existing task implementation
from .agents import rlinf_env_cfg
##
# Register Gym environments.
##

##
# Register Gym environments with RLinf support.
# Note: Using different ID from RLinf's registration to avoid conflict.
##

gym.register(
    id="Isaac-Install-Trocar-G129-Dex3-RLinf-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": rlinf_env_cfg.G1InstallTrocarEnvCfg,
        # RLinf agent configuration for VLA post-training
        "rlinf_cfg_entry_point": f"{agents.__name__}:RLinfInstallTrocarPPOCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Install-Trocar-G129-Dex3-RLinf-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": rlinf_env_cfg.G1InstallTrocarEvalEnvCfg,
        # RLinf agent configuration for VLA post-training
        "rlinf_cfg_entry_point": f"{agents.__name__}:RLinfInstallTrocarPPOCfg",
    },
    disable_env_checker=True,
)
