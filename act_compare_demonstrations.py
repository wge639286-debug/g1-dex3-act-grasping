#!/usr/bin/env python3
"""Compare ACT chunks with synchronized demonstration actions, without simulation."""
import argparse
import csv
import json
from pathlib import Path

import av
import numpy as np

from act_inference_smoke_test import DEFAULT_MODEL_DIR, DEFAULT_EPISODE_DIR, load_policy, predict_chunk


def select_frames(actions, cup, fps):
    # Index/middle flexion is positive. Thumb signs differ between joints.
    flexion = actions[:, 10:14].mean(axis=1)
    active = np.flatnonzero(flexion > 0.03)
    if not len(active):
        raise ValueError("No index/middle closure above 0.03 rad in episode")
    onset = int(active[0])
    halfway = np.flatnonzero((np.arange(len(actions)) >= onset) &
                             (flexion >= 0.5 * flexion.max()))
    # Peak center height identifies a late recorded pose; it is NOT a success test.
    return {"approach": max(0, onset - round(0.5 * fps)),
            "closure_onset": onset, "mid_closure": int(halfway[0]),
            "peak_recorded_height": int(np.argmax(cup[:, 2]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "act_demo_comparison")
    args = parser.parse_args()
    paths = sorted(args.episode_dir.expanduser().glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(args.episode_dir)
    out = args.output_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    policy, pre, post = load_policy(args.model_dir)
    rows, chunks = [], {}
    for episode_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as d:
            states, actions, cup = d["observation_state"], d["action"], d["cup_pose"]
            fps = float(d["fps"])
            video = path.parent / str(d["head_rgb_video"].item())
            joints = d["joint_names"].tolist()
        if states.shape != actions.shape or states.shape[1] != 14:
            raise ValueError(f"Invalid shape: {path}")
        frames = select_frames(actions, cup, fps)
        images = {}
        with av.open(str(video)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i in frames.values():
                    images[i] = frame.to_ndarray(format="rgb24")
                if i >= max(frames.values()):
                    break
        for phase, index in frames.items():
            policy.reset()
            predicted = predict_chunk(policy, pre, post, {
                "observation.state": states[index].astype(np.float32),
                "observation.images.head": images[index],
            }).squeeze(0).cpu().numpy()
            if predicted.shape != (policy.config.chunk_size, 14) or not np.isfinite(predicted).all():
                raise RuntimeError("Invalid prediction")
            # ACT action_delta_indices starts at 0. Exclude episode-end padding.
            reference = actions[index:index + len(predicted)]
            valid = len(reference)
            key = f"ep{episode_index:02d}_{phase}"
            chunks[key + "_predicted"] = predicted
            chunks[key + "_reference"] = reference
            mean_flexion = predicted[:, 10:14].mean(axis=1)
            closed = np.flatnonzero(mean_flexion >= 0.10)
            ref_closed = np.flatnonzero(reference[:, 10:14].mean(axis=1) >= 0.10)
            row = dict(episode=path.name, episode_index=episode_index, phase=phase,
                       frame=index, time_s=index / fps, valid_future_steps=valid,
                       hand_mae_first5=float(np.abs(predicted[:min(5, valid), 7:] - reference[:5, 7:]).mean()),
                       hand_mae_valid_chunk=float(np.abs(predicted[:valid, 7:] - reference[:, 7:]).mean()),
                       predicted_flexion_first5=float(mean_flexion[:min(5, valid)].mean()),
                       reference_flexion_first5=float(reference[:5, 10:14].mean()),
                       predicted_flexion_max100=float(mean_flexion.max()),
                       predicted_first_0_10rad_step=int(closed[0]) if len(closed) else -1,
                       reference_first_0_10rad_step=int(ref_closed[0]) if len(ref_closed) else -1)
            for j in range(7, 14):
                row[joints[j] + "_predicted_first5"] = float(predicted[:min(5, valid), j].mean())
                row[joints[j] + "_reference_first5"] = float(reference[:5, j].mean())
            rows.append(row)
        print(f"Compared {episode_index + 1}/{len(paths)}: {path.name}", flush=True)
    with (out / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / "chunks.npz", **chunks)
    summary = {}
    for phase in frames:
        subset = [r for r in rows if r["phase"] == phase]
        fields = ["hand_mae_first5", "hand_mae_valid_chunk", "predicted_flexion_first5",
                  "reference_flexion_first5", "predicted_flexion_max100"]
        summary[phase] = {k: float(np.mean([r[k] for r in subset])) for k in fields}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved comparison and complete chunks: {out.resolve()}")


if __name__ == "__main__":
    main()
