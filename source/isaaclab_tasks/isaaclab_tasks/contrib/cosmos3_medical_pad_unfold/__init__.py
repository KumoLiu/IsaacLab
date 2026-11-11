# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 medical-pad unfolding with Cosmos3 visual dynamics and classifier rewards."""

from gymnasium.envs.registration import register

register(
    id="Isaac-H2-MedicalPad-Unfold-Cosmos3-Direct-v0",
    entry_point="isaaclab_contrib.neural.neural_env:NeuralDirectEnv",
    disable_env_checker=True,
)
