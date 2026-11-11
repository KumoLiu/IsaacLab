# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic H2 wrist/fingertip geometry and action normalization contracts."""

import hashlib
import json

import numpy as np
import pytest

from isaaclab_tasks.contrib.cosmos3_medical_pad_unfold import h2_action_codec as module


@pytest.fixture
def codec(tmp_path):
    joints = []
    for side in ("left", "right"):
        wrist = f"{side}_hand_C_MC"
        joints.append(
            f'<joint name="{side}_shoulder_pitch_joint" type="revolute">'
            f'<parent link="torso_link"/><child link="{wrist}"/><axis xyz="0 0 1"/></joint>'
        )
        for index, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            joints.append(
                f'<joint name="{side}_{finger}_fixed" type="fixed">'
                f'<parent link="{wrist}"/><child link="{side}_{finger}_fingertip"/>'
                f'<origin xyz="{(index + 1) / 100} 0 0"/></joint>'
            )
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="analytic_h2">' + "".join(joints) + "</robot>")
    return module.H2UnifiedCodec(urdf, hashlib.sha256(urdf.read_bytes()).hexdigest())


@pytest.mark.parametrize("anchor_angle", [0, np.pi / 4])
def test_explicit_anchor_and_wrist_local_fingertips(codec, anchor_angle):
    commands = np.zeros((16, 58))
    commands[:, 7] = np.pi / 2
    anchor = np.zeros(58)
    anchor[7] = anchor_angle
    raw = codec.raw59(commands, observed_start_joints=anchor)
    angle = np.pi / 2 - anchor_angle
    rotation_columns = [np.cos(angle), np.sin(angle), 0, -np.sin(angle), np.cos(angle), 0]
    np.testing.assert_allclose(raw[:, 12:18], np.tile(rotation_columns, (16, 1)), atol=1e-14)
    np.testing.assert_allclose(raw[:, 37:43], np.tile([1, 0, 0, 0, 1, 0], (16, 1)), atol=1e-14)
    tips = np.zeros((5, 3))
    tips[:, 1] = np.arange(1, 6) / 100
    np.testing.assert_allclose(raw[:, 18:33], np.tile(tips.ravel(), (16, 1)), atol=1e-14)
    np.testing.assert_allclose(raw[:, 43:58], np.tile(-tips.ravel(), (16, 1)), atol=1e-14)
    assert not raw[:, np.r_[0:12, 33:37, 58]].any()


def test_normalization_padding_and_resource_integrity(codec, tmp_path, monkeypatch):
    stats = tmp_path / "global_asinh.json"
    stats.write_text(json.dumps({"global": {"q01": [-1] * 59, "q99": [1] * 59}}))
    monkeypatch.setattr(module, "STATS_SHA256", hashlib.sha256(stats.read_bytes()).hexdigest())
    actions = codec.for_inference(np.zeros((16, 58)), observed_start_joints=np.zeros(58), global_stats_path=stats)
    assert actions.shape == (16, 64) and actions.dtype == np.float32
    np.testing.assert_array_equal(actions[:, 12:18], np.tile([1, 0, 0, 0, 1, 0], (16, 1)))
    np.testing.assert_allclose(actions[:, 19], np.arcsinh(0.01) / np.arcsinh(1))
    assert not actions[:, np.r_[0:9, 33, 58:64]].any()
    stats.write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        codec.for_inference(np.zeros((16, 58)), observed_start_joints=np.zeros(58), global_stats_path=stats)


def test_invalid_joint_commands_are_rejected(codec):
    with pytest.raises(ValueError, match="Non-finite"):
        codec.raw59(np.full((16, 58), np.nan), observed_start_joints=np.zeros(58))
