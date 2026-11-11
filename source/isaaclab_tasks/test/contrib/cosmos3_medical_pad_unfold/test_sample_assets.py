# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bundled samples work without external datasets, meshes or model weights."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from isaaclab_tasks.contrib.cosmos3_medical_pad_unfold import h2_action_codec


def test_bundled_reset_and_replay_assets_are_portable():
    assets = Path(h2_action_codec.__file__).with_name("assets")
    provenance = json.loads((assets / "provenance.json").read_text())
    for name, identity in provenance["files"].items():
        path = assets / name
        assert path.resolve().is_relative_to(assets.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
    resets = json.loads((assets / "resets.json").read_text())
    assert [row["split"] for row in resets] == ["train", "validation"]
    for row in resets:
        assert not Path(row["image"]).is_absolute()
        with Image.open(assets / row["image"]) as image:
            assert image.mode == "RGB" and image.size == (640, 480)
        assert np.asarray(row["state"]).shape == (58,)
        assert np.isfinite(row["state"]).all()
    replay = json.loads((assets / "replay_actions.json").read_text())
    assert replay["action_space"] == "normalized_unified_v1"
    assert replay["domain_names"] == "hand_pose"
    assert np.asarray(replay["action"]).shape == (64, 59)
    assert np.isfinite(replay["action"]).all()
    config = yaml.safe_load((assets / "classifier.yaml").read_text())
    assert config["num_classes"] == 5 and config["reward_values"] == [-2, -1, 0, 1, 2]
    assert "model_path" not in config


def test_bundled_fk_needs_no_meshes_or_external_resources():
    assets = Path(h2_action_codec.__file__).with_name("assets")
    urdf = assets / "resources/H2_with_sharpa_hands.urdf"
    assert not ET.parse(urdf).getroot().findall(".//mesh")
    codec = h2_action_codec.H2UnifiedCodec(urdf, hashlib.sha256(urdf.read_bytes()).hexdigest())
    state = np.array(json.loads((assets / "resets.json").read_text())[1]["state"])
    controls = codec.for_inference(
        np.repeat(state[None], 16, axis=0),
        observed_start_joints=state,
        global_stats_path=assets / "resources/global_asinh.json",
    )
    assert controls.shape == (16, 64) and np.isfinite(controls).all()
