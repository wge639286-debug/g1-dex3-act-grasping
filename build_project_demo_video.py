#!/usr/bin/env python3
"""Build a concise portfolio demo from existing ACT rollout videos.

Requires OpenCV from the local ``lerobot`` environment.  The generated video
has no audio and does not rerun MuJoCo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
WIDTH, HEIGHT, FPS = 1280, 720, 25.0
BACKGROUND = (24, 27, 32)


@dataclass(frozen=True)
class Clip:
    path: Path
    start_s: float
    end_s: float | None
    caption: str


CLIPS = [
    Clip(
        PROJECT_DIR / "act_window_comparison.mp4",
        3.0,
        16.0,
        "Ablation: execute 5 actions vs 25 actions before replanning",
    ),
    Clip(
        PROJECT_DIR / "act_position_400_-075.mp4",
        8.0,
        None,
        "Closed-loop grasp: cup at (0.400, -0.075) m",
    ),
    Clip(
        PROJECT_DIR / "act_position_400_-050_30s.mp4",
        18.0,
        None,
        "Hard case: cup at (0.400, -0.050) m; success at 26.76 s",
    ),
    Clip(
        PROJECT_DIR / "act_position_400_-025_30s.mp4",
        9.0,
        None,
        "Closed-loop grasp: cup at (0.400, -0.025) m",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "g1_white_cup_act_demo.mp4",
    )
    return parser.parse_args()


def add_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (240, 240, 240),
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def title_card(lines: list[tuple[str, float]], seconds: float) -> list[np.ndarray]:
    frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    total_height = sum(62 if scale >= 1 else 48 for _, scale in lines)
    y = (HEIGHT - total_height) // 2
    for text, scale in lines:
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        x = max(32, (WIDTH - size[0]) // 2)
        add_text(frame, text, (x, y), scale)
        y += 62 if scale >= 1 else 48
    return [frame] * int(round(seconds * FPS))


def fit_frame(frame: np.ndarray) -> np.ndarray:
    available_height = HEIGHT - 94
    scale = min(WIDTH / frame.shape[1], available_height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    x = (WIDTH - resized.shape[1]) // 2
    y = 78 + (available_height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def iter_clip(clip: Clip):
    if not clip.path.is_file():
        raise FileNotFoundError(clip.path)
    capture = cv2.VideoCapture(str(clip.path))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or source_frames <= 0:
        raise ValueError(f"Cannot read video metadata: {clip.path}")
    duration = source_frames / source_fps
    end_s = min(clip.end_s if clip.end_s is not None else duration, duration)
    if not 0 <= clip.start_s < end_s:
        raise ValueError(f"Invalid time range for {clip.path.name}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(clip.start_s * source_fps))
    output_frames = int(round((end_s - clip.start_s) * FPS))
    for output_index in range(output_frames):
        source_index = round(
            clip.start_s * source_fps + output_index * source_fps / FPS
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = capture.read()
        if not ok:
            break
        canvas = fit_frame(frame)
        add_text(canvas, clip.caption, (34, 48), 0.72)
        yield canvas
    capture.release()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")

    cards_and_clips = [
        title_card(
            [
                ("Unitree G1 + Dex3 White-Cup Grasping", 1.05),
                ("MuJoCo | LeRobot ACT | Closed-loop vision policy", 0.72),
            ],
            2.5,
        ),
        title_card(
            [
                ("30 demonstrations | 12,956 frames | 25 Hz", 0.82),
                ("14D state/action | 480 x 848 head RGB", 0.82),
                ("ACT checkpoint: 30,000 training steps", 0.82),
            ],
            2.5,
        ),
    ]
    try:
        for frames in cards_and_clips:
            for frame in frames:
                writer.write(frame)
        for clip in CLIPS:
            for frame in iter_clip(clip):
                writer.write(frame)
        for frame in title_card(
            [
                ("Fixed-position functional benchmark: 10 / 10 passed once", 0.82),
                ("Local robustness: 10 / 10 passed with +/-2 mm cup jitter", 0.72),
                ("Mean success time: 25.32 s | final rise: 18.50 mm", 0.66),
                ("Strict success: >=10 mm table clearance for >=1.0 s", 0.62),
            ],
            3.5,
        ):
            writer.write(frame)
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(output))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if frames <= 0 or fps <= 0:
        raise RuntimeError(f"Generated video could not be validated: {output}")
    print(f"Demo video: {output}")
    print(f"Resolution: {WIDTH}x{HEIGHT}; duration: {frames / fps:.2f} s; fps: {fps:.2f}")


if __name__ == "__main__":
    main()
