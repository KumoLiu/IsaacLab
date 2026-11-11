# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific H2 FK and unified-v1 action encoding, independent of policy SDKs."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

STATS_SHA256 = "ae695e2c643a5626b3c3564f00705aa3a83ab0da7640549c7340db71172594b0"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("right", "left")
VALID = np.zeros(59, dtype=bool)
VALID[9:33] = True
VALID[34:58] = True
# Columns are canonical wrist axes expressed in the native Sharpa C_MC frame.
# X: thumb-to-pinky, Z: wrist-to-fingers, Y = Z cross X; determinant +1.
WRIST_ALIGNMENT = {
    "right": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "left": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}


def _urdf_name(name):
    match = re.fullmatch(r"k(Left|Right)(.+)", name)
    return (match[1].lower() + "_" + re.sub(r"(?<!^)(?=[A-Z])", "_", match[2]).lower() + "_joint") if match else name


def _rotation(axis, angles):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angles = np.asarray(angles, dtype=np.float64)
    c, s = np.cos(angles)[..., None, None], np.sin(angles)[..., None, None]
    return c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * cross


def _origin_transform(element):
    result = np.eye(4)
    if element is None:
        return result
    xyz = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    r, p, y = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    result[:3, :3] = _rotation([0, 0, 1], y) @ _rotation([0, 1, 0], p) @ _rotation([1, 0, 0], r)
    result[:3, 3] = xyz
    return result


class _H2FK:
    def __init__(self, xml, joint_names, *, root="torso_link"):
        self.xml = xml
        self.sha256 = hashlib.sha256(xml.encode()).hexdigest()
        self.names = [_urdf_name(name) for name in joint_names]
        if len(self.names) != 58 or len(set(self.names)) != 58:
            raise ValueError("Require the exact 58 unique recorded joints")
        robot = ET.fromstring(xml)
        by_child = {j.find("child").get("link"): j for j in robot.findall("joint")}
        targets = [s + "_hand_C_MC" for s in SIDES] + [s + "_" + f + "_fingertip" for s in SIDES for f in FINGERS]
        needed = set()
        for target in targets:
            child = target
            while child != root:
                if child not in by_child:
                    raise ValueError(f"No kinematic path from {root} to {target}")
                joint = by_child[child]
                needed.add(joint.get("name"))
                child = joint.find("parent").get("link")
        selected = [j for j in robot.findall("joint") if j.get("name") in needed]
        self.joints, available = [], {root}
        while selected:
            ready = [j for j in selected if j.find("parent").get("link") in available]
            if not ready:
                raise ValueError("Cyclic or disconnected kinematics")
            for j in ready:
                if j.get("type") not in ("fixed", "revolute", "continuous"):
                    raise ValueError("Unsupported joint type")
                name = j.get("name")
                idx = self.names.index(name) if name in self.names else None
                if j.get("type") != "fixed" and idx is None:
                    raise ValueError(f"Unrecorded movable joint {name}; refusing arbitrary zero")
                axis = j.find("axis")
                self.joints.append(
                    (
                        name,
                        j.get("type"),
                        idx,
                        j.find("parent").get("link"),
                        j.find("child").get("link"),
                        _origin_transform(j.find("origin")),
                        np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" "),
                    )
                )
                available.add(j.find("child").get("link"))
                selected.remove(j)
        self.root = root

    def frames(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 58 or not np.isfinite(q).all():
            raise ValueError("Expected finite N x 58 joint values in radians")
        result = {self.root: np.broadcast_to(np.eye(4), (len(q), 4, 4))}
        for name, kind, idx, parent, child, origin, axis in self.joints:
            joint = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
            if kind != "fixed":
                joint[:, :3, :3] = _rotation(axis, q[:, idx])
            result[child] = result[parent] @ origin @ joint
        return result

    def wrist_and_tips(self, q):
        frames = self.frames(q)
        wrists, tips = [], []
        for side in SIDES:
            wrist = frames[side + "_hand_C_MC"].copy()
            wrist[:, :3, :3] = wrist[:, :3, :3] @ WRIST_ALIGNMENT[side]
            positions = np.stack([frames[side + "_" + finger + "_fingertip"][:, :3, 3] for finger in FINGERS], axis=1)
            local = np.einsum("nji,nkj->nki", wrist[:, :3, :3], positions - wrist[:, None, :3, 3])
            wrists.append(wrist)
            tips.append(local)
        return np.stack(wrists, axis=1), np.stack(tips, axis=1)


def _load_stats(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != STATS_SHA256:
        raise ValueError("Official normalization statistics digest mismatch")
    stats = json.loads(data)["global"]
    low, high = np.array(stats["q01"]), np.array(stats["q99"])
    return (low + high) / 2, np.maximum((high - low) / 2, 1e-8)


def _raw_window(anchor_wrists, target_wrists, target_tips):
    """Encode one 16-command window using an explicitly supplied past anchor.

    Anchor choice is recorded by the caller and must match training/inference.
    Future measured poses must never be supplied to this function.
    """
    anchor = np.asarray(anchor_wrists)
    target, fingertips = np.asarray(target_wrists), np.asarray(target_tips)
    if anchor.shape != (2, 4, 4) or target.shape != (16, 2, 4, 4) or fingertips.shape != (16, 2, 5, 3):
        raise ValueError("Expected 2 anchors and 16 dual-wrist/fingertip targets")
    raw = np.zeros((16, 59), dtype=np.float64)
    for side, wrist_start, tips_start in [(0, 9, 18), (1, 34, 43)]:
        a, t = anchor[side], target[:, side]
        relative_rotation = a[:3, :3].T @ t[:, :3, :3]
        relative_translation = (t[:, :3, 3] - a[:3, 3]) @ a[:3, :3]
        raw[:, wrist_start : wrist_start + 3] = relative_translation
        raw[:, wrist_start + 3 : wrist_start + 9] = relative_rotation[:, :, :2].transpose(0, 2, 1).reshape(16, 6)
        raw[:, tips_start : tips_start + 15] = fingertips[:, side].reshape(16, 15)
    if not np.isfinite(raw).all():
        raise ValueError("Invalid canonical geometry")
    return raw


def _normalize(raw, stats):
    offset, scale = stats
    out = np.arcsinh((raw - offset) / scale) / np.arcsinh(1.0)
    out[..., ~VALID] = 0
    return out


_ARM = ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "Elbow", "WristRoll", "WristPitch", "WristYaw")
_HAND = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
JOINT_NAMES = tuple(
    [f"k{s}{j}" for s in ("Left", "Right") for j in _ARM] + [f"{s}_{j}" for s in ("left", "right") for j in _HAND]
)


class H2UnifiedCodec:
    """Convert measured-anchor H2 joint commands [rad] into normalized unified actions.

    This task-owned codec reads kinematics and normalization resources only;
    it does not initialize a robot driver or another simulation backend.
    """

    def __init__(self, urdf_path: str | Path, urdf_sha256: str):
        xml = Path(urdf_path).read_bytes()
        if hashlib.sha256(xml).hexdigest() != urdf_sha256:
            raise ValueError("H2 URDF identity mismatch")
        self.fk = _H2FK(xml.decode(), JOINT_NAMES)

    def raw59(self, commands: np.ndarray, *, observed_start_joints: np.ndarray) -> np.ndarray:
        """Encode 16 joint targets relative to the explicitly supplied past anchor."""
        commands = np.asarray(commands, dtype=np.float64)
        observed = np.asarray(observed_start_joints, dtype=np.float64)
        if commands.shape != (16, 58) or observed.shape != (58,):
            raise ValueError("Expected 16 H2 commands and an explicit 58D observation-start anchor")
        if not np.isfinite(commands).all() or not np.isfinite(observed).all():
            raise ValueError("Non-finite H2 joints")
        anchor, _ = self.fk.wrist_and_tips(observed[None])
        wrists, tips = self.fk.wrist_and_tips(commands)
        return _raw_window(anchor[0], wrists, tips)

    def for_inference(
        self, commands: np.ndarray, *, observed_start_joints: np.ndarray, global_stats_path: str | Path
    ) -> np.ndarray:
        """Return normalized float32 actions with shape (16,64); inactive fields are zero."""
        # Deliberately no previous-command fallback. Live inference must supply
        # the current measured start state; offline users must declare their
        # anchor conditioning rather than silently changing this contract.
        raw = self.raw59(commands, observed_start_joints=observed_start_joints)
        result = np.pad(_normalize(raw, _load_stats(global_stats_path)), ((0, 0), (0, 5))).astype(np.float32)
        assert np.isfinite(result).all() and not result[:, np.r_[~VALID, np.ones(5, dtype=bool)]].any()
        return result
