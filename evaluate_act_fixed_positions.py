#!/usr/bin/env python3
"""Evaluate one ACT checkpoint once at ten deterministic cup positions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY = (
    PROJECT_DIR
    / "models"
    / "g1_white_cup_14d_act_030000"
    / "pretrained_model"
)
DEFAULT_OUTPUT = PROJECT_DIR / "act_fixed_position_eval"

# Baseline position followed by a 3x3 grid, inset 10 mm from the selected
# region boundaries. These are fixed so checkpoints and controller settings
# can be compared on exactly the same inputs.
POSITIONS = [(0.360, -0.075)] + [
    (x, y)
    for x in (0.340, 0.370, 0.400)
    for y in (-0.075, -0.050, -0.025)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execution-steps", type=int, default=25)
    parser.add_argument("--rollout-seconds", type=float, default=20.0)
    return parser.parse_args()


def read_diagnostics(path: Path) -> dict[str, float | int | bool]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty diagnostics: {path}")

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows])

    finger_targets = np.stack(
        [
            values(f"right_hand_{finger}_{joint}_joint_requested_rad")
            for finger in ("index", "middle")
            for joint in (0, 1)
        ],
        axis=1,
    )
    loaded = np.stack(
        [values(f"{finger}_normal_force_N") > 0.1 for finger in ("thumb", "index", "middle")],
        axis=1,
    ).all(axis=1)
    clear_hold = values("clear_hold_s")
    return {
        "success": bool(clear_hold.max() >= 1.0 - 1e-9),
        "duration_s": float(values("time_s")[-1]),
        "max_clear_hold_s": float(clear_hold.max()),
        "max_lowest_clearance_mm": float(values("cup_clearance_mm").max()),
        "final_lowest_clearance_mm": float(values("cup_clearance_mm")[-1]),
        "final_table_contacts": int(values("cup_table_contacts")[-1]),
        "max_index_middle_target_rad": float(finger_targets.max()),
        "three_finger_loaded_fraction": float(loaded.mean()),
    }


def save_results(output_dir: Path, results: list[dict]) -> None:
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "execution_steps": results[0]["execution_steps"],
        "completed": len(results),
        "successes": sum(row["success"] for row in results),
        "success_rate": sum(row["success"] for row in results) / len(results),
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    policy = args.policy.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not policy.is_dir():
        raise FileNotFoundError(policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "egl"

    for index, (cup_x, cup_y) in enumerate(POSITIONS):
        stem = output_dir / f"position_{index:02d}"
        log_path = stem.with_suffix(".log")
        command = [
            sys.executable,
            str(PROJECT_DIR / "g1_cup_minimal.py"),
            "--act-policy",
            str(policy),
            "--act-execution-steps",
            str(args.execution_steps),
            "--act-rollout-seconds",
            str(args.rollout_seconds),
            "--act-cup-x",
            str(cup_x),
            "--act-cup-y",
            str(cup_y),
            "--act-video-out",
            str(stem.with_suffix(".mp4")),
            "--act-no-video",
        ]
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=PROJECT_DIR,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"Position {index} failed with exit code {process.returncode}; see {log_path}"
            )
        diagnostic_path = stem.with_suffix(".diagnostics.csv")
        result = {
            "position_index": index,
            "cup_x_m": cup_x,
            "cup_y_m": cup_y,
            "execution_steps": args.execution_steps,
            **read_diagnostics(diagnostic_path),
            "wall_seconds": time.perf_counter() - started,
        }
        results.append(result)
        save_results(output_dir, results)
        print(
            f"Position {index + 1}/{len(POSITIONS)}: "
            f"({cup_x:.3f}, {cup_y:.3f}) -> "
            f"{'PASS' if result['success'] else 'FAIL'}, "
            f"clearance={result['max_lowest_clearance_mm']:.2f} mm, "
            f"finger target={result['max_index_middle_target_rad']:.3f} rad",
            flush=True,
        )

    print(
        f"Fixed-position evaluation complete: "
        f"{sum(row['success'] for row in results)}/{len(results)} PASS"
    )
    print(f"Results: {output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
