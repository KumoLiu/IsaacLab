# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T data config module for IsaacLab RLinf integration.

This module provides customizable GR00T data configurations that users can
modify for their specific embodiment/task without changing gr00t or rlinf code.

Usage:
    # In run.sh, set:
    export RLINF_DATA_CONFIG="policy.gr00t_config"
    export RLINF_DATA_CONFIG_CLASS="policy.gr00t_config:IsaacLabDataConfig"
"""

from .gr00t_config import IsaacLabDataConfig

__all__ = ["IsaacLabDataConfig"]
