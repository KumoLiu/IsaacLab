# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf PPO configuration for the Assemble Trocar task (G129 + Dex3).

This mirrors the pattern of ``rsl_rl_ppo_cfg.py`` files used by RSL-RL tasks,
but targets the RLinf + GR00T pipeline instead.  All observation/action mapping
parameters are defined here as a single source of truth.
"""

from isaaclab_rl.rlinf.rl_cfg import (
    RLinfActionMappingCfg,
    RLinfGR00TMappingCfg,
    RLinfGR00TStateMappingCfg,
    RLinfGR00TVideoMappingCfg,
    RLinfIsaacLabCfg,
    RLinfStateSpecCfg,
)


class AssembleTrocarRLinfPPOCfg(RLinfIsaacLabCfg):
    """IsaacLab → RLinf → GR00T observation/action mapping for Assemble Trocar."""

    # -- Task description for language-conditioned models ----------------------
    task_description: str = "install trocar from box"

    # -- IsaacLab → RLinf observation mapping ----------------------------------
    # main_images: single camera key for the main view
    main_images: str = "front_camera"

    # extra_view_images: list of camera keys to stack as (B, N, H, W, C)
    extra_view_images: list[str] = [  # noqa: RUF012
        "left_wrist_camera",
        "right_wrist_camera",
    ]

    # states: list of state specs with optional slicing
    states: list[RLinfStateSpecCfg] = [  # noqa: RUF012
        RLinfStateSpecCfg(key="robot_joint_state", slice=(15, 29)),  # G129 shoulder joints
        RLinfStateSpecCfg(key="robot_dex3_joint_state"),  # full tensor
    ]

    # -- RLinf → GR00T format conversion ---------------------------------------
    gr00t_mapping: RLinfGR00TMappingCfg = RLinfGR00TMappingCfg(
        video=RLinfGR00TVideoMappingCfg(
            main_images="video.room_view",
            extra_view_images=["video.left_wrist_view", "video.right_wrist_view"],
        ),
        # Slice concatenated states into GR00T state keys
        # Total states: 14 (shoulder) + 14 (dex3) = 28 dims
        state=[
            RLinfGR00TStateMappingCfg(gr00t_key="state.left_arm", slice=(0, 7)),
            RLinfGR00TStateMappingCfg(gr00t_key="state.right_arm", slice=(7, 14)),
            RLinfGR00TStateMappingCfg(gr00t_key="state.left_hand", slice=(14, 21)),
            RLinfGR00TStateMappingCfg(gr00t_key="state.right_hand", slice=(21, 28)),
        ],
    )

    # -- GR00T → IsaacLab action conversion ------------------------------------
    action_mapping: RLinfActionMappingCfg = RLinfActionMappingCfg(
        prefix_pad=15,  # Pad zeros at front for G129 body joints (not controlled)
        suffix_pad=0,
    )

    # -- GR00T model / embodiment configuration --------------------------------
    obs_converter_type: str = "dex3"
    embodiment_tag: str = "new_embodiment"
    embodiment_tag_id: int = 31
    data_config_class: str = "policy.gr00t_config:IsaacLabDataConfig"
