# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-free G1 trocar task using a DreamDojo neural simulator."""

from gymnasium.envs.registration import register

register(
    id="Isaac-DreamDojo-Trocar-Direct-v0",
    entry_point="isaaclab_contrib.neural.neural_env:NeuralDirectEnv",
    disable_env_checker=True,
)
