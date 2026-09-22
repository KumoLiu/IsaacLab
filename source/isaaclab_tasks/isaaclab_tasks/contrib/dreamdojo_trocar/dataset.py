# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recorded G1 trocar resets and action replay from a LeRobot v2.1 tree.

Reset uses image/state frame 2 and historical action/state frame 0 at 30 Hz.
Only the standalone replay demo reads subsequent real frames, for comparison.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_recording(dataset: Path, episode: int, chunks: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read exact 30-Hz commands and time-aligned reference frames; never pad."""
    metadata = json.loads((dataset / "meta/info.json").read_text())
    modality = json.loads((dataset / "meta/modality.json").read_text())
    if float(metadata["fps"]) != 30:
        raise ValueError("The existing G1 WM bridge requires a 30-Hz recording")
    if episode < 0 or episode >= int(metadata["total_episodes"]) or chunks < 1:
        raise ValueError("Invalid episode index or chunk count")
    keys = {"episode_index": episode, "episode_chunk": episode // int(metadata["chunks_size"])}
    parquet = dataset / metadata["data_path"].format(**keys)
    video_key = modality["video"]["head_view"]["original_key"]
    video = dataset / metadata["video_path"].format(**keys, video_key=video_key)
    table = pq.read_table(parquet, columns=["action", "timestamp", "frame_index"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    end_frame = 2 + 12 * chunks
    if actions.ndim != 2 or actions.shape[1] != 28 or not np.isfinite(actions).all():
        raise ValueError("Expected finite 28-D recorded actions")
    if len(actions) <= end_frame:
        raise ValueError(f"Episode has {len(actions)} rows; need at least {end_frame + 1}. Reduce --chunks")
    if not np.array_equal(table["frame_index"].to_numpy(), np.arange(len(actions))):
        raise ValueError("Frame indices must be contiguous and start at zero")
    np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(len(actions)) / 30, atol=1e-5)
    with imageio.get_reader(video) as reader:
        frames = np.stack([reader.get_data(i) for i in range(2, end_frame + 1, 2)])
    if frames.shape[1:] != (480, 640, 3):
        raise ValueError(f"Expected 480x640 RGB recordings, got {frames.shape}")
    provenance = {
        "parquet": str(parquet),
        "video": str(video),
        "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "first_image_frame_30hz": 2,
        "first_action_frame_30hz": 2,
        "last_image_frame_30hz": end_frame,
        "reference_usage": "comparison_only_after_reset; never sent to step",
    }
    return actions[2:end_frame].reshape(chunks, 12, 28), frames, provenance


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class LeRobotV21InitDataset(Dataset):
    """Load one fixed, causally aligned reset per recorded episode.

    Args:
        data_path: LeRobot v2.1 root containing ``meta/``, ``data/`` and ``videos/``.
        image_size: Output ``(H, W)``; ``None`` preserves the recorded resolution.

    Items contain an RGB ``image`` tensor [3, H, W] in [0, 1], 28-D ``state``,
    ``previous_state`` and ``previous_action`` tensors in the recording's joint
    units, plus ``task``, ``episode_index`` and ``start_frame``. The head camera
    and frame-2 reset match the retained WM checkpoint; no KIR or mixing is used.
    """

    def __init__(self, data_path: str | Path, image_size: tuple[int, int] | None = (480, 640)) -> None:
        self.root = Path(data_path).expanduser().resolve()
        self.image_size = image_size
        self.info = json.loads((self.root / "meta/info.json").read_text())
        modality = json.loads((self.root / "meta/modality.json").read_text())
        if float(self.info["fps"]) != 30:
            raise ValueError("The existing G1 WM bridge requires a 30-Hz recording")
        self.video_original_key = modality["video"]["head_view"].get("original_key", "head_view")
        self.task_texts = {
            int(row["task_index"]): str(row["task"]) for row in _read_jsonl(self.root / "meta/tasks.jsonl")
        }
        self.episode_tasks = {
            int(row["episode_index"]): str(row["tasks"][0])
            for row in _read_jsonl(self.root / "meta/episodes.jsonl")
            if row.get("tasks")
        }
        missing = [index for index in range(len(self)) if not self._parquet_path(index).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} episodes missing parquet files, first few: {missing[:5]}")

    def __len__(self) -> int:
        return int(self.info["total_episodes"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        table = pq.read_table(self._parquet_path(index))
        if table.num_rows < 3:
            raise ValueError(f"Episode {index} needs at least 3 frames for the frame-2 reset")
        vectors = {}
        for column in ("observation.state", "action"):
            if column not in table.column_names:
                raise ValueError(f"Episode {index} is missing required column {column}")
            values = np.asarray(table[column].slice(0, 3).to_pylist(), dtype=np.float32)
            if values.shape != (3, 28) or not np.isfinite(values).all():
                raise ValueError(f"Episode {index}: {column} must contain finite 28-D vectors")
            vectors[column] = torch.from_numpy(values)

        task = self.episode_tasks.get(index)
        if task is None:
            task = (
                self.task_texts.get(int(table["task_index"][0].as_py()), "")
                if "task_index" in table.column_names
                else ""
            )
        image = self._decode_video_frames(self._video_path(index), [2])[0]
        return {
            "image": self._prepare_image(image),
            "state": vectors["observation.state"][2],
            "previous_state": vectors["observation.state"][0],
            "previous_action": vectors["action"][0],
            "task": task,
            "episode_index": index,
            "start_frame": 2,
        }

    def _parquet_path(self, episode_index: int) -> Path:
        chunk = episode_index // int(self.info.get("chunks_size", 1000))
        rel = self.info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
        return (self.root / rel).resolve()

    def _video_path(self, episode_index: int) -> Path:
        chunk = episode_index // int(self.info.get("chunks_size", 1000))
        rel = self.info["video_path"].format(
            episode_chunk=chunk, video_key=self.video_original_key, episode_index=episode_index
        )
        return (self.root / rel).resolve()

    def _decode_video_frames(self, video_path: Path, frame_indices: list[int]) -> np.ndarray:
        """Decode exact frame indices; a short video must not silently repeat its last frame."""
        import decord

        reader = decord.VideoReader(str(video_path), num_threads=1)
        return reader.get_batch(frame_indices).asnumpy()

    def _prepare_image(self, frame_hwc_uint8: np.ndarray) -> torch.Tensor:
        """Convert an RGB uint8 frame to a [3, H, W] float tensor in [0, 1]."""
        image = torch.from_numpy(frame_hwc_uint8).permute(2, 0, 1).float() / 255.0
        if self.image_size is not None and tuple(image.shape[1:]) != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(0)
        return image
