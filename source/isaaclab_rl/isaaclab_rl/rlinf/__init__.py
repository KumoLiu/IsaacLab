# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf integration for IsaacLab.

This module provides the extension mechanism for integrating IsaacLab tasks
with RLinf's distributed RL training framework for VLA models like GR00T.

Configuration Classes
---------------------

The module exposes :class:`RLinfIsaacLabCfg` and its sub-configs so that task
authors can define structured, validated RLinf configurations — the same way
RSL-RL tasks define :class:`~isaaclab_rl.rsl_rl.rl_cfg.RslRlOnPolicyRunnerCfg`.

.. code-block:: python

    from isaaclab_rl.rlinf import RLinfIsaacLabCfg, RLinfStateSpecCfg

    class MyTaskRLinfCfg(RLinfIsaacLabCfg):
        task_description = "pick up the box"
        main_images = "front_camera"
        states = [RLinfStateSpecCfg(key="joint_pos")]

Extension Module
----------------

The extension module (``extension.py``) is loaded by RLinf via the
``RLINF_EXT_MODULE`` environment variable and handles:

1. Registering IsaacLab tasks into RLinf's ``REGISTER_ISAACLAB_ENVS``
2. Registering GR00T obs/action converters
3. Patching GR00T ``get_model`` for custom embodiments

Usage:
    .. code-block:: bash

        export RLINF_EXT_MODULE="isaaclab_rl.rlinf.extension"
        export RLINF_ISAACLAB_TASKS="Isaac-MyTask-v0"
"""

from .rl_cfg import (
    RLinfActionMappingCfg,
    RLinfGR00TMappingCfg,
    RLinfGR00TStateMappingCfg,
    RLinfGR00TVideoMappingCfg,
    RLinfIsaacLabCfg,
    RLinfStateSpecCfg,
)

__all__ = [
    "RLinfActionMappingCfg",
    "RLinfGR00TMappingCfg",
    "RLinfGR00TStateMappingCfg",
    "RLinfGR00TVideoMappingCfg",
    "RLinfIsaacLabCfg",
    "RLinfStateSpecCfg",
]
