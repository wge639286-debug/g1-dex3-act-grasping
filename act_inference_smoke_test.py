#!/usr/bin/env python3
"""Run one recorded G1 observation through a local ACT checkpoint on CPU."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import av
import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import prepare_observation_for_inference


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "g1_white_cup_14d_act_030000"
    / "pretrained_model"
)
DEFAULT_EPISODE_DIR = PROJECT_DIR / "demonstrations_npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that a recorded RGB/state observation produces a finite ACT action chunk."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--episode",
        type=Path,
        default=None,
        help="Source episode NPZ. By default, use the first episode in demonstrations_npz.",
    )
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=1,
        help="Number of timed forward passes after one warm-up pass.",
    )
    return parser.parse_args()


def choose_episode(path: Path | None) -> Path:
    if path is not None:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    episodes = sorted(DEFAULT_EPISODE_DIR.glob("episode_*.npz"))
    if not episodes:
        raise FileNotFoundError(f"No episode_*.npz files found in {DEFAULT_EPISODE_DIR}")
    return episodes[0]


def scalar(episode: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(episode[key])
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar, got {value.shape}")
    return value.item()


def decode_rgb_frame(video_path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"Video {video_path} has no frame {frame_index}")


def load_recorded_observation(
    episode_path: Path, frame_index: int
) -> tuple[dict[str, np.ndarray], np.ndarray, Path]:
    with np.load(episode_path, allow_pickle=False) as episode:
        states = np.asarray(episode["observation_state"], dtype=np.float32)
        actions = np.asarray(episode["action"], dtype=np.float32)
        if not 0 <= frame_index < len(states):
            raise IndexError(f"frame-index {frame_index} is outside [0, {len(states) - 1}]")
        video_path = episode_path.parent / str(scalar(episode, "head_rgb_video"))
        state = states[frame_index].copy()
        recorded_action = actions[frame_index].copy()

    rgb = decode_rgb_frame(video_path, frame_index)
    if rgb.shape != (480, 848, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Unexpected RGB frame: shape={rgb.shape}, dtype={rgb.dtype}")
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"Unexpected state: shape={state.shape}, finite={np.isfinite(state).all()}")

    return {
        "observation.state": state,
        "observation.images.head": rgb,
    }, recorded_action, video_path


def load_policy(model_dir: Path) -> tuple[ACTPolicy, object, object]:
    model_dir = model_dir.expanduser().resolve()
    config = ACTConfig.from_pretrained(model_dir, local_files_only=True)
    config.device = "cpu"
    # The checkpoint already contains the trained backbone. Avoid re-downloading
    # ImageNet initialization weights while constructing the model locally.
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, preprocessor, postprocessor


def predict_chunk(
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    observation: dict[str, np.ndarray],
) -> torch.Tensor:
    prepared = prepare_observation_for_inference(
        {key: value.copy() for key, value in observation.items()},
        torch.device("cpu"),
        task="Pick up the white cup",
        robot_type="g1_dex3",
    )
    prepared = preprocessor(prepared)
    with torch.inference_mode():
        normalized_chunk = policy.predict_action_chunk(prepared)
        chunk = postprocessor(normalized_chunk)
    return chunk


def main() -> None:
    args = parse_args()
    if args.timed_runs < 1:
        raise ValueError("--timed-runs must be at least 1")

    episode_path = choose_episode(args.episode)
    observation, recorded_action, video_path = load_recorded_observation(
        episode_path, args.frame_index
    )

    load_start = time.perf_counter()
    policy, preprocessor, postprocessor = load_policy(args.model_dir)
    load_seconds = time.perf_counter() - load_start

    warmup_chunk = predict_chunk(policy, preprocessor, postprocessor, observation)
    timings = []
    predicted_chunk = warmup_chunk
    for _ in range(args.timed_runs):
        start = time.perf_counter()
        predicted_chunk = predict_chunk(policy, preprocessor, postprocessor, observation)
        timings.append(time.perf_counter() - start)

    expected_shape = (1, policy.config.chunk_size, 14)
    if tuple(predicted_chunk.shape) != expected_shape:
        raise RuntimeError(
            f"Predicted chunk shape {tuple(predicted_chunk.shape)}, expected {expected_shape}"
        )
    if not torch.isfinite(predicted_chunk).all():
        raise RuntimeError("Predicted action chunk contains non-finite values")

    chunk = predicted_chunk.squeeze(0).cpu().numpy()
    first_action = chunk[0]
    print(f"Model: {Path(args.model_dir).expanduser().resolve()}")
    print(f"Episode: {episode_path}")
    print(f"Video: {video_path.name}; frame={args.frame_index}; RGB={observation['observation.images.head'].shape}")
    print(f"State: shape={observation['observation.state'].shape}, finite={np.isfinite(observation['observation.state']).all()}")
    print(f"Policy load time: {load_seconds:.3f} s on CPU")
    print(f"Action chunk: shape={chunk.shape}, finite={np.isfinite(chunk).all()}")
    print(
        "Timed inference: "
        f"mean={np.mean(timings):.3f} s, min={np.min(timings):.3f} s, "
        f"max={np.max(timings):.3f} s over {len(timings)} run(s)"
    )
    print(f"First predicted action: {np.array2string(first_action, precision=5)}")
    print(f"Recorded action at frame: {np.array2string(recorded_action, precision=5)}")
    print(f"First-action L1 difference: {np.mean(np.abs(first_action - recorded_action)):.6f} rad")
    print("ACT inference smoke test PASS")


if __name__ == "__main__":
    main()
