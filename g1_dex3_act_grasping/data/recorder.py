"""Episode compression, discovery, and cup-region persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compress_recorded_episode(
    timestamps: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    fps: float,
    *,
    leading_seconds: float = 0.5,
    trailing_seconds: float = 1.0,
    idle_threshold_seconds: float = 0.6,
    idle_keep_seconds: float = 0.2,
) -> dict[str, np.ndarray | int]:
    """Trim an episode and shorten long runs where action and state are still."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if (
        timestamps.ndim != 1
        or states.shape != (timestamps.size, 14)
        or actions.shape != (timestamps.size, 14)
        or timestamps.size < 2
        or not np.isfinite(timestamps).all()
        or not np.isfinite(states).all()
        or not np.isfinite(actions).all()
        or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError("invalid timestamps/state/action arrays")

    dynamic = np.zeros(timestamps.size, dtype=bool)
    action_step = np.max(np.abs(np.diff(actions, axis=0)), axis=1)
    state_step = np.max(np.abs(np.diff(states, axis=0)), axis=1)
    if not np.any(action_step > 1e-6):
        raise ValueError("episode contains no action changes")
    dynamic[1:] = (action_step > 1e-6) | (state_step > 0.01 / fps)
    changed_indices = np.flatnonzero(dynamic)

    leading_frames = max(0, round(leading_seconds * fps))
    trailing_frames = max(0, round(trailing_seconds * fps))
    start = max(0, int(changed_indices[0]) - leading_frames)
    stop = min(timestamps.size, int(changed_indices[-1]) + 1 + trailing_frames)

    changed = dynamic[start:stop]
    keep = np.ones(stop - start, dtype=bool)
    threshold_frames = max(1, round(idle_threshold_seconds * fps))
    kept_idle_frames = max(2, round(idle_keep_seconds * fps))
    index = 0
    while index < changed.size:
        if changed[index]:
            index += 1
            continue
        run_start = index
        while index < changed.size and not changed[index]:
            index += 1
        run_stop = index
        run_length = run_stop - run_start
        # Runs at either cropped boundary are the requested leading/trailing
        # context and are deliberately left intact.
        if (
            run_start > 0
            and run_stop < changed.size
            and run_length > threshold_frames
            and run_length > kept_idle_frames
        ):
            keep_left = (kept_idle_frames + 1) // 2
            keep_right = kept_idle_frames // 2
            keep[run_start + keep_left:run_stop - keep_right] = False

    original_indices = np.arange(start, stop, dtype=np.int64)[keep]
    frame_index = np.arange(original_indices.size, dtype=np.int64)
    return {
        "timestamp": frame_index.astype(np.float64) / fps,
        "sim_timestamp": timestamps[start:stop][keep],
        "frame_index": frame_index,
        "original_frame_index": original_indices,
        "observation_state": states[start:stop][keep],
        "action": actions[start:stop][keep],
        "original_frame_count": int(timestamps.size),
        "trim_start_frame": int(start),
    }


def find_recorded_cup_positions(record_dir: Path) -> list[np.ndarray]:
    """Return cup XY positions from complete, replayable episodes."""
    positions: list[np.ndarray] = []
    if not record_dir.is_dir():
        return positions
    for path in sorted(record_dir.glob("episode_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as episode:
                if not {"cup_pose", "cup_initial_position", "format_version"}.issubset(
                    episode.files
                ):
                    continue
                if int(episode["format_version"]) < 3:
                    continue
                position = np.asarray(
                    episode["cup_initial_position"], dtype=np.float64
                )
                if position.shape == (3,) and np.isfinite(position).all():
                    positions.append(position[:2].copy())
        except (OSError, ValueError):
            print(f"Warning: ignored unreadable episode while counting: {path}")
    return positions


def count_rgb_episodes(record_dir: Path) -> int:
    """Count complete format-v4 episodes whose referenced RGB video exists."""
    count = 0
    if not record_dir.is_dir():
        return count
    for path in sorted(record_dir.glob("episode_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as episode:
                if not {"format_version", "images_included", "head_rgb_video"}.issubset(
                    episode.files
                ):
                    continue
                if int(episode["format_version"]) < 4:
                    continue
                if not bool(episode["images_included"]):
                    continue
                video_name = str(episode["head_rgb_video"])
                video_path = path.parent / video_name
                if video_path.is_file() and video_path.stat().st_size > 0:
                    count += 1
        except (OSError, ValueError):
            print(f"Warning: ignored unreadable episode while counting RGB: {path}")
    return count


def load_cup_region(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        region = {
            key: float(payload[key])
            for key in ("x_min", "x_max", "y_min", "y_max")
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid cup region file {path}: {exc}") from exc
    if not (
        np.isfinite(list(region.values())).all()
        and region["x_min"] < region["x_max"]
        and region["y_min"] < region["y_max"]
    ):
        raise ValueError(f"Invalid cup region bounds in {path}: {region}")
    return region
