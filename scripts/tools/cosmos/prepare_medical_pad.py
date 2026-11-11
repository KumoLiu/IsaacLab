# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare repository-local model assets and RLinf config for the medical-pad example."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import yaml


def main() -> None:
    """Check or download declared artifacts, prepare model resources, and write the run config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing files with declared HF sources")
    parser.add_argument("--replay-only", action="store_true", help="Only require WM, video VAE and text tokenizer")
    parser.add_argument("--smoke", action="store_true", help="Generate a 64-frame/16-batch integration-check config")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    task = root / "source/isaaclab_tasks/isaaclab_tasks/contrib/cosmos3_medical_pad_unfold"
    assets = task / "assets"
    manifest = json.loads((assets / "artifacts.json").read_text())
    models = root / manifest["model_directory"]
    models.mkdir(parents=True, exist_ok=True)
    cache = root / ".pretrained_checkpoints/huggingface/hub"
    names = ("world_model", "vae", "tokenizer") if args.replay_only else tuple(manifest["artifacts"])
    unavailable = [
        name
        for name in names
        if not manifest["artifacts"][name]["repo_id"]
        and any(not (models / name / filename).is_file() for filename in manifest["artifacts"][name]["files"])
    ]
    if args.download and unavailable:
        parser.error(f"No download source configured for: {', '.join(unavailable)}")
    missing = []
    for name in names:
        artifact = manifest["artifacts"][name]
        for filename in artifact["files"]:
            target = models / name / filename
            if not target.is_file() and args.download and artifact["repo_id"]:
                from huggingface_hub import hf_hub_download

                source = hf_hub_download(
                    repo_id=artifact["repo_id"],
                    revision=artifact["revision"],
                    filename="/".join(filter(None, (artifact["prefix"], filename))),
                    cache_dir=cache,
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            if not target.is_file() or not target.stat().st_size:
                missing.append(str(target.relative_to(root)))
                continue
            if not target.resolve().is_relative_to(root):
                parser.error(f"Model assets must be inside this checkout, not external symlinks: {target}")
            expected = artifact.get("sha256", {}).get(filename)
            if expected:
                with target.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        parser.error(f"Artifact identity mismatch: {target}")

    # The source recipe contains portable defaults; generated absolute paths belong
    # to this checkout and do not depend on the author's model/data directories.
    cfg = yaml.safe_load((task / "agents/isaaclab_cosmos3_grpo_medical_pad_unfold.yaml").read_text())
    cfg["actor"]["model"]["model_path"] = str(models / "policy")
    cfg["actor"]["model"]["backbone_model_path"] = str(models / "backbone")
    for split in ("train", "eval"):
        env = cfg["env"][split]
        settings = env["isaaclab"]
        settings["reset_manifest"] = str(assets / "resets.json")
        settings["codec_resources"] = str(assets / "resources")
        settings["reward"]["checkpoint"] = str(models / "classifier/reward_model.pt")
        settings["reward"]["config_file"] = str(assets / "classifier.yaml")
        settings["world_model"]["checkpoint_path"] = str(models / "world_model")
        if args.smoke:
            env["max_episode_steps"] = env["max_steps_per_rollout_epoch"] = 64
    if args.smoke:
        cfg["actor"]["global_batch_size"] = 16
    config_name = "medical_pad_unfold_smoke" if args.smoke else "medical_pad_unfold"
    config_path = models / f"{config_name}.yaml"
    if not config_path.exists():
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    wm_config = models / "world_model/config.json"
    support = (models / "vae/Wan2.2_VAE.pth", models / "tokenizer/tokenizer_config.json")
    if wm_config.is_file() and all(path.is_file() for path in support):
        spec = importlib.util.find_spec("cosmos_framework")
        if spec is None:
            parser.error("Install the cosmos3 or cosmos3-rl extra before preparing WM resources")
        reasoner = Path(spec.origin).parent / "model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
        destination = models / "world_model/resources" / reasoner.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reasoner, destination)
        config = json.loads(wm_config.read_text())

        def resolve_resources(value):
            if isinstance(value, dict):
                if "vae_path" in value:
                    value["vae_path"] = str(support[0])
                if value.get("_target") == "qwen3_vl_mot_config_from_json_file":
                    value["json_file"] = str(destination)
                if value.get("_target") == "create_qwen2_tokenizer_with_download":
                    value["pretrained_model_name"] = str(models / "tokenizer")
                for item in value.values():
                    resolve_resources(item)
            elif isinstance(value, list):
                for item in value:
                    resolve_resources(item)
            elif (
                isinstance(value, str) and Path(value).is_absolute() and not Path(value).resolve().is_relative_to(root)
            ):
                parser.error(f"Unresolved external path in WM config: {value}")

        resolve_resources(config)
        wm_config.write_text(json.dumps(config, indent=2) + "\n")
    policy_config = models / "policy/config.json"
    if policy_config.is_file() and not args.replay_only:
        config = json.loads(policy_config.read_text())
        config["use_flash_attention"] = False
        policy_config.write_text(json.dumps(config, indent=2) + "\n")
    status = {
        "mode": "replay" if args.replay_only else "rl",
        "config": str(config_path.relative_to(root)),
        "missing_files": missing,
        "unconfigured_downloads": [name for name in names if not manifest["artifacts"][name]["repo_id"]],
    }
    (models / "asset_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if missing:
        parser.exit(1, "Missing model files. Place the files listed in assets/artifacts.json in the model directories.\n")
    if args.replay_only:
        print("Replay assets ready. Run scripts/demos/cosmos3_replay.py with the prepared world model.")
    else:
        print("RL assets ready. Use the generated config directory with isaaclab train/play --rl_library rlinf.")


if __name__ == "__main__":
    main()
