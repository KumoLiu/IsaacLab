# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf integration for IsaacLab.

This module provides the extension mechanism for integrating IsaacLab tasks
with RLinf's distributed RL training framework for VLA models like GR00T.

Usage:
    Set the following environment variables before running RLinf:
    
    # Enable the extension
    export RLINF_EXT_MODULE="isaaclab_rl.rlinf.extension"
    
    # Specify tasks to register
    export RLINF_ISAACLAB_TASKS="Isaac-MyTask-v0"
    
    # Configure observation mapping (JSON format)
    export RLINF_OBS_MAP_JSON='{"main_images":"camera","states":["joint_pos"]}'
    
    # Configure GR00T model patch (optional)
    export RLINF_EMBODIMENT_TAG="new_embodiment"
    export RLINF_DATA_CONFIG_CLASS="policy.gr00t_config:MyDataConfig"

The extension module (extension.py) handles:
    1. Registering IsaacLab tasks into RLinf's REGISTER_ISAACLAB_ENVS
    2. Registering GR00T obs/action converters
    3. Patching GR00T get_model for custom embodiments
"""

# The main functionality is in extension.py, which is loaded by RLinf
# via RLINF_EXT_MODULE environment variable.
