#!/usr/bin/env python3
"""Convert G1 cup demonstrations from format-v4 NPZ files to LeRobot v0.6.x."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_DIR / "demonstrations_npz"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "lerobot_dataset" / "g1_white_cup"
DEFAULT_REPO_ID = "local/g1_white_cup"
CAMERA_KEY = "observation.images.head"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert synchronized G1 NPZ/MP4 episodes into a local LeRobot dataset."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", default="Pick up the white cup")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N episodes (useful for a smoke test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before conversion.",
    )
    return parser.parse_args()


def scalar(episode: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(episode[key])
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar, got shape {value.shape}")
    return value.item()


def inspect_episode(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as episode:
        required = {
            "timestamp",
            "observation_state",
            "action",
            "cup_pose",
            "fps",
            "joint_names",
            "images_included",
            "head_rgb_video",
            "head_rgb_shape",
            "format_version",
        }
        missing = sorted(required - set(episode.files))
        if missing:
            raise ValueError(f"{path.name}: missing fields {missing}")

        if int(scalar(episode, "format_version")) < 4:
            raise ValueError(f"{path.name}: format_version must be at least 4")
        if not bool(scalar(episode, "images_included")):
            raise ValueError(f"{path.name}: images_included is false")

        timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
        states = np.asarray(episode["observation_state"])
        actions = np.asarray(episode["action"])
        cup_poses = np.asarray(episode["cup_pose"], dtype=np.float64)
        joint_names = [str(name) for name in episode["joint_names"].tolist()]
        fps = float(scalar(episode, "fps"))
        frame_count = len(timestamps)

        expected_shapes = {
            "observation_state": (frame_count, 14),
            "action": (frame_count, 14),
            "cup_pose": (frame_count, 7),
        }
        for name, array in (
            ("observation_state", states),
            ("action", actions),
            ("cup_pose", cup_poses),
        ):
            if array.shape != expected_shapes[name]:
                raise ValueError(
                    f"{path.name}: {name} shape {array.shape}, expected {expected_shapes[name]}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{path.name}: {name} contains non-finite values")

        if frame_count < 2 or not np.isfinite(timestamps).all():
            raise ValueError(f"{path.name}: invalid timestamps")
        expected_dt = 1.0 / fps
        if not np.allclose(np.diff(timestamps), expected_dt, atol=1e-6, rtol=0.0):
            raise ValueError(f"{path.name}: timestamps are not uniformly sampled at {fps:g} Hz")
        if len(joint_names) != 14:
            raise ValueError(f"{path.name}: expected 14 joint names, got {len(joint_names)}")

        video_name = str(scalar(episode, "head_rgb_video"))
        video_path = path.parent / video_name
        if not video_path.is_file():
            raise FileNotFoundError(f"{path.name}: video does not exist: {video_path}")

        rgb_shape = tuple(int(value) for value in episode["head_rgb_shape"])
        if rgb_shape != (frame_count, 480, 848, 3):
            raise ValueError(
                f"{path.name}: head_rgb_shape {rgb_shape}, expected {(frame_count, 480, 848, 3)}"
            )

        return {
            "path": path,
            "video_path": video_path,
            "frame_count": frame_count,
            "fps": fps,
            "joint_names": joint_names,
            "initial_cup_position": cup_poses[0, :3].tolist(),
            "peak_cup_rise_mm": float((cup_poses[:, 2].max() - cup_poses[0, 2]) * 1000.0),
            "final_cup_rise_mm": float((cup_poses[-1, 2] - cup_poses[0, 2]) * 1000.0),
        }


def load_numeric_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as episode:
        states = np.asarray(episode["observation_state"], dtype=np.float32)
        actions = np.asarray(episode["action"], dtype=np.float32)
    return states, actions


def build_features(joint_names: list[str]) -> dict:
    axes = {"axes": joint_names}
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": axes,
        },
        CAMERA_KEY: {
            "dtype": "video",
            "shape": (480, 848, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": axes,
        },
    }


def add_episode(dataset: LeRobotDataset, info: dict, task: str) -> None:
    states, actions = load_numeric_arrays(info["path"])
    decoded = 0
    with av.open(str(info["video_path"])) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != (848, 480):
            raise ValueError(
                f"{info['video_path'].name}: video size is {stream.width}x{stream.height}, expected 848x480"
            )
        if float(stream.average_rate) != info["fps"]:
            raise ValueError(
                f"{info['video_path'].name}: video fps is {stream.average_rate}, expected {info['fps']:g}"
            )

        for decoded, video_frame in enumerate(container.decode(video=0), start=1):
            frame_index = decoded - 1
            if frame_index >= info["frame_count"]:
                raise ValueError(
                    f"{info['video_path'].name}: video has more than {info['frame_count']} frames"
                )
            dataset.add_frame(
                {
                    "observation.state": states[frame_index],
                    CAMERA_KEY: video_frame.to_ndarray(format="rgb24"),
                    "action": actions[frame_index],
                    "task": task,
                }
            )

    if decoded != info["frame_count"]:
        raise ValueError(
            f"{info['video_path'].name}: decoded {decoded} frames, expected {info['frame_count']}"
        )
    dataset.save_episode()


def verify_dataset(root: Path, repo_id: str, expected_episodes: int, expected_frames: int) -> None:
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    if dataset.num_episodes != expected_episodes:
        raise RuntimeError(
            f"LeRobot verification: {dataset.num_episodes} episodes, expected {expected_episodes}"
        )
    if len(dataset) != expected_frames:
        raise RuntimeError(f"LeRobot verification: {len(dataset)} frames, expected {expected_frames}")

    first = dataset[0]
    if tuple(first["observation.state"].shape) != (14,):
        raise RuntimeError("LeRobot verification: observation.state shape is not (14,)")
    if tuple(first["action"].shape) != (14,):
        raise RuntimeError("LeRobot verification: action shape is not (14,)")
    if tuple(first[CAMERA_KEY].shape) != (3, 480, 848):
        raise RuntimeError(
            f"LeRobot verification: {CAMERA_KEY} shape is {tuple(first[CAMERA_KEY].shape)}"
        )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = sorted(input_dir.glob("episode_*.npz"))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No episode_*.npz files found in {input_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Choose another path or pass --overwrite."
            )
        shutil.rmtree(output_dir)

    print(f"Inspecting {len(paths)} source episodes...")
    infos = [inspect_episode(path) for path in paths]
    fps_values = {info["fps"] for info in infos}
    joint_name_values = {tuple(info["joint_names"]) for info in infos}
    if fps_values != {25.0}:
        raise ValueError(f"Expected all episodes at 25 Hz, got {sorted(fps_values)}")
    if len(joint_name_values) != 1:
        raise ValueError("Joint names differ between episodes")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    total_frames = sum(info["frame_count"] for info in infos)
    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=temp_dir,
            fps=25,
            robot_type="unitree_g1_dex3",
            features=build_features(infos[0]["joint_names"]),
            use_videos=True,
            streaming_encoding=True,
            encoder_threads=2,
        )
        for episode_index, info in enumerate(infos):
            print(
                f"[{episode_index + 1:02d}/{len(infos):02d}] {info['path'].name}: "
                f"{info['frame_count']} frames"
            )
            add_episode(dataset, info, args.task)
        dataset.finalize()

        manifest = {
            "source_format": "g1_cup_minimal format v4",
            "repo_id": args.repo_id,
            "fps": 25,
            "task": args.task,
            "camera_key": CAMERA_KEY,
            "episodes": [
                {
                    "source_npz": info["path"].name,
                    "source_video": info["video_path"].name,
                    "frames": info["frame_count"],
                    "initial_cup_position": info["initial_cup_position"],
                    "peak_cup_rise_mm": info["peak_cup_rise_mm"],
                    "final_cup_rise_mm": info["final_cup_rise_mm"],
                }
                for info in infos
            ],
        }
        (temp_dir / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        verify_dataset(temp_dir, args.repo_id, len(infos), total_frames)
        temp_dir.rename(output_dir)
    except Exception:
        if dataset is not None and dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(
        f"Conversion PASS: episodes={len(infos)}, frames={total_frames}, "
        f"duration={total_frames / 25.0:.2f} s"
    )
    print(f"LeRobot dataset: {output_dir}")


if __name__ == "__main__":
    main()
