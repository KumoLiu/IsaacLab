# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for the RLinf integration with IsaacLab.

This module defines structured configuration classes that mirror
`RSL-RL's rl_cfg.py <isaaclab_rl.rsl_rl.rl_cfg>`_ pattern.  They provide:

* Type-checked, documented fields with sensible defaults.
* Standard ``dataclasses`` for lightweight, dependency-free configs.
* A single source of truth for every parameter the extension reads.

The top-level class that task authors should subclass or instantiate is
:class:`RLinfIsaacLabCfg`.  It maps directly to the ``env.train.isaaclab``
section of the Hydra YAML consumed by RLinf.

Example – Python-first (recommended)
-------------------------------------

.. code-block:: python

    from isaaclab_rl.rlinf.rl_cfg import RLinfIsaacLabCfg, RLinfStateSpecCfg

    class MyTaskCfg(RLinfIsaacLabCfg):
        task_description = "pick up the box"
        main_images = "front_camera"
        extra_view_images = ["wrist_camera"]
        states = [
            RLinfStateSpecCfg(key="joint_pos"),
            RLinfStateSpecCfg(key="joint_vel", slice=(0, 6)),
        ]

Example – YAML-first (backward compatible)
-------------------------------------------

The extension can still read a Hydra YAML file.  Use
:func:`RLinfIsaacLabCfg.from_yaml_dict` to construct a validated config
from the raw dict returned by ``yaml.safe_load()``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field


# ---------------------------------------------------------------------------
# Observation-mapping sub-configs
# ---------------------------------------------------------------------------


@dataclass
class RLinfStateSpecCfg:
    """Specification for a single state tensor that should be extracted from
    IsaacLab's policy observation and concatenated into the RLinf ``states``
    vector.

    If :attr:`slice` is ``None`` the full tensor is used; otherwise only the
    columns ``[start, end)`` are kept.
    """

    key: str = MISSING
    """Key in the IsaacLab ``policy`` observation dict."""

    slice: tuple[int, int] | None = None
    """Optional column slice ``(start, end)``.  ``None`` → use full tensor."""


@dataclass
class RLinfGR00TVideoMappingCfg:
    """Maps IsaacLab camera keys to GR00T video keys."""

    main_images: str = "video.room_view"
    """GR00T key for the main camera image."""

    extra_view_images: list[str] = field(
        default_factory=lambda: ["video.left_wrist_view", "video.right_wrist_view"]
    )
    """GR00T keys corresponding to the extra cameras (same order as
    :attr:`RLinfIsaacLabCfg.extra_view_images`)."""


@dataclass
class RLinfGR00TStateMappingCfg:
    """Maps a slice of the concatenated state vector to a GR00T state key."""

    gr00t_key: str = MISSING
    """GR00T state key, e.g. ``"state.left_arm"``."""

    slice: tuple[int, int] = MISSING
    """Column slice ``(start, end)`` into the concatenated state vector."""


@dataclass
class RLinfGR00TMappingCfg:
    """Defines the complete IsaacLab → GR00T format conversion."""

    video: RLinfGR00TVideoMappingCfg = field(default_factory=RLinfGR00TVideoMappingCfg)
    """Video key mapping."""

    state: list[RLinfGR00TStateMappingCfg] = field(default_factory=list)
    """State key mapping with slicing.  Each entry maps a range of the
    concatenated state vector to a GR00T state key."""


@dataclass
class RLinfActionMappingCfg:
    """Defines how GR00T actions are padded/mapped back to IsaacLab actions."""

    prefix_pad: int = 0
    """Number of zero-columns prepended to the action vector.

    Useful when the first *N* joints of a robot (e.g. body joints) are not
    controlled by the policy.
    """

    suffix_pad: int = 0
    """Number of zero-columns appended to the action vector."""


# ---------------------------------------------------------------------------
# Top-level IsaacLab config for RLinf
# ---------------------------------------------------------------------------


@dataclass
class RLinfIsaacLabCfg:
    """IsaacLab-specific configuration consumed by the RLinf extension.

    This class captures every parameter that ``extension.py`` needs to

    * convert IsaacLab observations into the RLinf / GR00T format,
    * convert GR00T actions back into the IsaacLab format,
    * register custom embodiment tags and obs/action converters.

    An instance of this class maps 1-to-1 to the ``env.train.isaaclab``
    section of the Hydra YAML.
    """

    # -- IsaacLab → RLinf observation mapping ----------------------------------

    task_description: str = ""
    """Natural-language task description for language-conditioned models."""

    main_images: str = ""
    """Key in ``camera_images`` for the main camera view.

    Set to an empty string if the task does not provide camera images.
    """

    extra_view_images: list[str] = field(default_factory=list)
    """Keys in ``camera_images`` for extra cameras (stacked to
    ``(B, N, H, W, C)``)."""

    states: list[RLinfStateSpecCfg] = field(default_factory=list)
    """State observation specs.  Each entry is extracted from the ``policy``
    observation dict and concatenated along the last dim to produce a single
    ``(B, D)`` state tensor."""

    # -- RLinf → GR00T format conversion ---------------------------------------

    gr00t_mapping: RLinfGR00TMappingCfg = field(default_factory=RLinfGR00TMappingCfg)
    """Mapping from the RLinf intermediate format to GR00T model inputs."""

    action_mapping: RLinfActionMappingCfg = field(default_factory=RLinfActionMappingCfg)
    """Mapping from GR00T action output back to IsaacLab actions."""

    # -- Model / embodiment config ---------------------------------------------

    obs_converter_type: str = "isaaclab"
    """Converter type name registered in RLinf's ``simulation_io`` registry."""

    embodiment_tag: str = "new_embodiment"
    """GR00T embodiment tag string."""

    embodiment_tag_id: int = 31
    """Numeric ID for the embodiment tag in RLinf's ``EMBODIMENT_TAG_MAPPING``."""

    data_config_class: str = ""
    """Module path to a GR00T ``DataConfig`` subclass, e.g.
    ``"policy.gr00t_config:IsaacLabDataConfig"``.

    If empty, RLinf's default ``get_model`` is used without patching.
    """

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def from_yaml_dict(raw: dict) -> RLinfIsaacLabCfg:
        """Construct an :class:`RLinfIsaacLabCfg` from a raw YAML dict.

        This handles the heterogeneous ``states`` list (plain strings and dicts)
        that cannot be handled automatically by :meth:`from_dict`.

        Args:
            raw: The ``env.train.isaaclab`` section parsed from YAML.

        Returns:
            A fully populated configuration instance.
        """
        cfg = RLinfIsaacLabCfg()

        # --- simple scalar / list fields ---
        for field_name in (
            "task_description",
            "main_images",
            "extra_view_images",
            "obs_converter_type",
            "embodiment_tag",
            "embodiment_tag_id",
            "data_config_class",
        ):
            if field_name in raw:
                setattr(cfg, field_name, raw[field_name])

        # --- states: list[str | dict] → list[RLinfStateSpecCfg] ---
        raw_states = raw.get("states", [])
        cfg.states = []
        for entry in raw_states:
            if isinstance(entry, str):
                cfg.states.append(RLinfStateSpecCfg(key=entry))
            elif isinstance(entry, dict):
                spec = RLinfStateSpecCfg(key=entry["key"])
                if "slice" in entry and entry["slice"] is not None:
                    spec.slice = tuple(entry["slice"])
                cfg.states.append(spec)

        # --- gr00t_mapping ---
        raw_gm = raw.get("gr00t_mapping", {})
        if raw_gm:
            raw_video = raw_gm.get("video", {})
            cfg.gr00t_mapping.video.main_images = raw_video.get(
                "main_images", cfg.gr00t_mapping.video.main_images
            )
            cfg.gr00t_mapping.video.extra_view_images = raw_video.get(
                "extra_view_images", cfg.gr00t_mapping.video.extra_view_images
            )

            raw_state = raw_gm.get("state", [])
            cfg.gr00t_mapping.state = []
            for s in raw_state:
                cfg.gr00t_mapping.state.append(
                    RLinfGR00TStateMappingCfg(
                        gr00t_key=s["gr00t_key"],
                        slice=tuple(s["slice"]),
                    )
                )

        # --- action_mapping ---
        raw_am = raw.get("action_mapping", {})
        if raw_am:
            cfg.action_mapping.prefix_pad = raw_am.get(
                "prefix_pad", cfg.action_mapping.prefix_pad
            )
            cfg.action_mapping.suffix_pad = raw_am.get(
                "suffix_pad", cfg.action_mapping.suffix_pad
            )

        return cfg
