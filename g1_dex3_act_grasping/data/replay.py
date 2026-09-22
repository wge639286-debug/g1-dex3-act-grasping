"""Replay and synchronized video rendering for recorded episodes."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from g1_dex3_act_grasping.constants import IMAGE_HEIGHT, IMAGE_WIDTH


def replay_recorded_episode(
    episode_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_qpos_addresses: np.ndarray,
    cup_qpos_address: int,
) -> None:
    episode_path = episode_path.expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"Replay episode does not exist: {episode_path}")
    with np.load(episode_path, allow_pickle=False) as episode:
        required = {"timestamp", "observation_state", "cup_pose"}
        missing = sorted(required.difference(episode.files))
        if missing:
            raise ValueError(
                f"Replay episode is missing {missing}; old recordings without "
                "per-frame cup_pose cannot be replayed completely"
            )
        timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
        states = np.asarray(episode["observation_state"], dtype=np.float64)
        cup_poses = np.asarray(episode["cup_pose"], dtype=np.float64)
        fps = float(episode["fps"]) if "fps" in episode.files else 25.0
    if (
        timestamps.ndim != 1
        or timestamps.size == 0
        or states.shape != (timestamps.size, 14)
        or cup_poses.shape != (timestamps.size, 7)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(states).all()
        or not np.isfinite(cup_poses).all()
        or fps <= 0.0
    ):
        raise ValueError("Replay episode has invalid timestamps/state/cup_pose arrays")

    from mujoco import viewer as mj_viewer

    print(
        f"Replay: {episode_path}; frames={timestamps.size}, fps={fps:g}, "
        f"duration={timestamps[-1]:.3f} s"
    )
    print("Replay starts immediately and holds the final frame; close the viewer to exit.")
    with mj_viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=False
    ) as viewer:
        viewer.cam.lookat[:] = [0.33, -0.08, 0.90]
        viewer.cam.distance = 0.85
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -20
        period = 1.0 / fps
        for frame in range(timestamps.size):
            if not viewer.is_running():
                break
            frame_started = time.monotonic()
            with viewer.lock():
                data.qpos[controlled_qpos_addresses] = states[frame]
                data.qpos[cup_qpos_address:cup_qpos_address + 7] = cup_poses[frame]
                data.qvel[:] = 0.0
                data.time = float(timestamps[frame])
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(0.0, period - (time.monotonic() - frame_started)))
        if viewer.is_running():
            print("Replay finished; final frame is held.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    print("Replay viewer closed; exiting cleanly.")
    sys.stdout.flush()
    os._exit(0)


def render_episode_head_rgb(
    episode_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_qpos_addresses: np.ndarray,
    cup_qpos_address: int,
) -> Path:
    """Render frame-aligned head-camera video from a saved simulator trajectory."""
    import av
    from fractions import Fraction

    episode_path = episode_path.expanduser().resolve()
    with np.load(episode_path, allow_pickle=False) as episode:
        required = {"timestamp", "observation_state", "cup_pose", "fps"}
        missing = sorted(required.difference(episode.files))
        if missing:
            raise ValueError(f"Cannot render RGB; episode is missing {missing}")
        timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
        states = np.asarray(episode["observation_state"], dtype=np.float64)
        cup_poses = np.asarray(episode["cup_pose"], dtype=np.float64)
        fps = float(episode["fps"])
    if (
        timestamps.ndim != 1
        or timestamps.size == 0
        or states.shape != (timestamps.size, 14)
        or cup_poses.shape != (timestamps.size, 7)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(states).all()
        or not np.isfinite(cup_poses).all()
        or fps <= 0.0
    ):
        raise ValueError("Cannot render RGB; invalid timestamp/state/cup_pose arrays")

    output_path = episode_path.with_suffix(".head_rgb.mp4")
    temporary_path = output_path.with_name(output_path.stem + ".tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    container = None
    try:
        container = av.open(str(temporary_path), "w", format="mp4")
        stream = container.add_stream(
            "libx264", rate=Fraction(str(fps)).limit_denominator(1000)
        )
        stream.width = IMAGE_WIDTH
        stream.height = IMAGE_HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        camera_scene_option = mujoco.MjvOption()
        camera_scene_option.sitegroup[:] = 0
        with mujoco.Renderer(
            model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
        ) as renderer:
            for frame_index in range(timestamps.size):
                data.qpos[controlled_qpos_addresses] = states[frame_index]
                data.qpos[cup_qpos_address:cup_qpos_address + 7] = cup_poses[
                    frame_index
                ]
                data.qvel[:] = 0.0
                data.time = float(timestamps[frame_index])
                mujoco.mj_forward(model, data)
                renderer.update_scene(
                    data,
                    camera="head_camera",
                    scene_option=camera_scene_option,
                )
                rgb = renderer.render()
                if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or rgb.dtype != np.uint8:
                    raise RuntimeError(
                        f"Invalid rendered RGB frame: {rgb.shape}, {rgb.dtype}"
                    )
                video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                if (frame_index + 1) % 100 == 0:
                    print(
                        f"RGB render: {frame_index + 1}/{timestamps.size} frames",
                        flush=True,
                    )
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        temporary_path.replace(output_path)
    except Exception:
        if container is not None:
            container.close()
        temporary_path.unlink(missing_ok=True)
        raise
    print(
        f"RGB video SAVED: {output_path}; frames={timestamps.size}, "
        f"resolution={IMAGE_WIDTH}x{IMAGE_HEIGHT}, fps={fps:g}"
    )
    return output_path


def render_episode_review(
    episode_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_qpos_addresses: np.ndarray,
    cup_qpos_address: int,
) -> Path:
    """Create a synchronized third-person and head-camera review video."""
    import av
    import cv2
    from fractions import Fraction

    episode_path = episode_path.expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"Review episode does not exist: {episode_path}")
    with np.load(episode_path, allow_pickle=False) as episode:
        required = {"timestamp", "observation_state", "cup_pose", "fps"}
        missing = sorted(required.difference(episode.files))
        if missing:
            raise ValueError(f"Cannot review episode; missing {missing}")
        timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
        states = np.asarray(episode["observation_state"], dtype=np.float64)
        cup_poses = np.asarray(episode["cup_pose"], dtype=np.float64)
        fps = float(episode["fps"])
        video_name = (
            str(episode["head_rgb_video"])
            if "head_rgb_video" in episode.files
            else episode_path.with_suffix(".head_rgb.mp4").name
        )
    head_video_path = episode_path.parent / video_name
    if not head_video_path.is_file():
        raise FileNotFoundError(
            f"Head RGB video does not exist: {head_video_path}"
        )
    if (
        timestamps.ndim != 1
        or timestamps.size == 0
        or states.shape != (timestamps.size, 14)
        or cup_poses.shape != (timestamps.size, 7)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(states).all()
        or not np.isfinite(cup_poses).all()
        or fps <= 0.0
    ):
        raise ValueError("Cannot review episode; invalid trajectory arrays")

    output_path = episode_path.with_suffix(".review.mp4")
    temporary_path = output_path.with_name(output_path.stem + ".tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    head_container = av.open(str(head_video_path))
    head_stream = head_container.streams.video[0]
    if (
        head_stream.width != IMAGE_WIDTH
        or head_stream.height != IMAGE_HEIGHT
        or abs(float(head_stream.average_rate) - fps) > 1e-6
    ):
        head_container.close()
        raise ValueError(
            "Head RGB video resolution/FPS does not match the episode"
        )
    head_frames = head_container.decode(video=0)
    output_container = None
    try:
        output_container = av.open(str(temporary_path), "w", format="mp4")
        output_stream = output_container.add_stream(
            "libx264", rate=Fraction(str(fps)).limit_denominator(1000)
        )
        output_stream.width = 2 * IMAGE_WIDTH
        output_stream.height = IMAGE_HEIGHT
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"crf": "18", "preset": "medium"}

        third_person_camera = mujoco.MjvCamera()
        third_person_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        third_person_camera.lookat[:] = [0.36, -0.085, 0.90]
        third_person_camera.distance = 0.85
        third_person_camera.azimuth = -90
        third_person_camera.elevation = -20
        with mujoco.Renderer(
            model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
        ) as renderer:
            for frame_index in range(timestamps.size):
                try:
                    head_frame = next(head_frames)
                except StopIteration as exc:
                    raise ValueError(
                        f"Head RGB video ended before frame {frame_index}"
                    ) from exc
                head_rgb = head_frame.to_ndarray(format="rgb24")
                data.qpos[controlled_qpos_addresses] = states[frame_index]
                data.qpos[cup_qpos_address:cup_qpos_address + 7] = cup_poses[
                    frame_index
                ]
                data.qvel[:] = 0.0
                data.time = float(timestamps[frame_index])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=third_person_camera)
                third_rgb = renderer.render()
                combined = np.concatenate((third_rgb, head_rgb), axis=1)
                cv2.line(
                    combined,
                    (IMAGE_WIDTH, 0),
                    (IMAGE_WIDTH, IMAGE_HEIGHT - 1),
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    combined, "Third-person trajectory", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    combined, "Head camera", (IMAGE_WIDTH + 20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )
                video_frame = av.VideoFrame.from_ndarray(combined, format="rgb24")
                for packet in output_stream.encode(video_frame):
                    output_container.mux(packet)
                if (frame_index + 1) % 100 == 0:
                    print(
                        f"Review render: {frame_index + 1}/{timestamps.size} frames",
                        flush=True,
                    )
        if next(head_frames, None) is not None:
            raise ValueError("Head RGB video has more frames than the episode")
        for packet in output_stream.encode():
            output_container.mux(packet)
        output_container.close()
        output_container = None
        temporary_path.replace(output_path)
    except Exception:
        if output_container is not None:
            output_container.close()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        head_container.close()
    print(
        f"Review video SAVED: {output_path}; frames={timestamps.size}, "
        f"resolution={2 * IMAGE_WIDTH}x{IMAGE_HEIGHT}, fps={fps:g}"
    )
    return output_path
