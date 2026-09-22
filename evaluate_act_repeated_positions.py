#!/usr/bin/env python3
"""Repeat ACT closed-loop evaluation with optional cup-position jitter."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evaluate_act_fixed_positions import POSITIONS, read_diagnostics


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY = (
    PROJECT_DIR / "models" / "g1_white_cup_14d_act_030000" / "pretrained_model"
)
DEFAULT_OUTPUT = PROJECT_DIR / "act_repeated_position_eval"
DEFAULT_JITTER_OUTPUT = PROJECT_DIR / "act_position_jitter_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to act_repeated_position_eval without "
            "jitter and act_position_jitter_eval with jitter."
        ),
    )
    parser.add_argument("--execution-steps", type=int, default=25)
    parser.add_argument("--rollout-seconds", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--cup-jitter-mm",
        type=float,
        default=0.0,
        help="Independent uniform +/- X/Y offset around each nominal position.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260922,
        help="Base seed used to reproduce cup offsets (default: 20260922).",
    )
    parser.add_argument(
        "--position-index",
        type=int,
        action="append",
        dest="position_indices",
        help=(
            "Evaluate only this zero-based position index; repeat the option to "
            "select several. Omit it to evaluate all ten positions."
        ),
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save every trial MP4. Diagnostics CSV files are always saved.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Refuse to reuse existing trial diagnostics in the output directory.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the generated cup coordinates and exit without running MuJoCo.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not 1 <= args.rollout_seconds <= 120:
        parser.error("--rollout-seconds must be between 1 and 120")
    if args.execution_steps < 1:
        parser.error("--execution-steps must be at least 1")
    if not 0 <= args.cup_jitter_mm <= 50:
        parser.error("--cup-jitter-mm must be between 0 and 50")
    if args.position_indices is not None:
        invalid = [i for i in args.position_indices if not 0 <= i < len(POSITIONS)]
        if invalid:
            parser.error(f"invalid --position-index values: {invalid}")
    return args


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_existing_results(
    path: Path,
    execution_steps: int,
    rollout_seconds: float,
    cup_jitter_mm: float,
    seed: int,
) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict] = []
    for raw in raw_rows:
        if int(raw["execution_steps"]) != execution_steps or not math.isclose(
            float(raw["rollout_seconds"]), rollout_seconds
        ):
            raise ValueError(
                f"{path} contains a different protocol; choose a new --output-dir"
            )
        saved_jitter = float(raw.get("cup_jitter_mm", 0.0))
        saved_seed = int(raw.get("random_seed", seed))
        if not math.isclose(saved_jitter, cup_jitter_mm) or (
            cup_jitter_mm > 0 and saved_seed != seed
        ):
            raise ValueError(
                f"{path} contains different jitter/seed settings; choose a new --output-dir"
            )
        row = dict(raw)
        for key in ("position_index", "trial_index", "execution_steps", "final_table_contacts"):
            row[key] = int(row[key])
        row["random_seed"] = saved_seed
        for key in (
            "cup_x_m", "cup_y_m", "rollout_seconds", "duration_s",
            "max_clear_hold_s", "max_lowest_clearance_mm",
            "final_lowest_clearance_mm", "max_index_middle_target_rad",
            "three_finger_loaded_fraction", "final_cup_rise_mm", "wall_seconds",
        ):
            row[key] = float(row[key])
        row["cup_jitter_mm"] = saved_jitter
        row["nominal_cup_x_m"] = float(raw.get("nominal_cup_x_m", row["cup_x_m"]))
        row["nominal_cup_y_m"] = float(raw.get("nominal_cup_y_m", row["cup_y_m"]))
        row["offset_x_mm"] = float(raw.get("offset_x_mm", 0.0))
        row["offset_y_mm"] = float(raw.get("offset_y_mm", 0.0))
        row["success"] = str(row["success"]).lower() == "true"
        rows.append(row)
    return rows


def upsert(rows: list[dict], result: dict) -> None:
    key = (result["position_index"], result["trial_index"])
    rows[:] = [
        row for row in rows
        if (row["position_index"], row["trial_index"]) != key
    ]
    rows.append(result)


def trial_position(
    nominal_x: float,
    nominal_y: float,
    jitter_mm: float,
    seed: int,
    position_index: int,
    trial_index: int,
) -> tuple[float, float, float, float]:
    trial_seed = seed + position_index * 1_000_003 + trial_index * 97
    rng = random.Random(trial_seed)
    offset_x_mm = rng.uniform(-jitter_mm, jitter_mm)
    offset_y_mm = rng.uniform(-jitter_mm, jitter_mm)
    return (
        nominal_x + offset_x_mm / 1000.0,
        nominal_y + offset_y_mm / 1000.0,
        offset_x_mm,
        offset_y_mm,
    )


def aggregate(rows: list[dict]) -> tuple[list[dict], dict]:
    position_rows: list[dict] = []
    for position_index, (cup_x, cup_y) in enumerate(POSITIONS):
        trials = [r for r in rows if r["position_index"] == position_index]
        if not trials:
            continue
        successful = [r for r in trials if r["success"]]
        durations = [float(r["duration_s"]) for r in successful]
        position_rows.append(
            {
                "position_index": position_index,
                "cup_x_m": cup_x,
                "cup_y_m": cup_y,
                "trials": len(trials),
                "successes": len(successful),
                "success_rate": len(successful) / len(trials),
                "mean_success_time_s": statistics.fmean(durations) if durations else "",
                "std_success_time_s": statistics.pstdev(durations) if durations else "",
                "min_success_time_s": min(durations) if durations else "",
                "max_success_time_s": max(durations) if durations else "",
                "mean_final_cup_rise_mm": statistics.fmean(
                    float(r["final_cup_rise_mm"]) for r in trials
                ),
                "min_actual_x_m": min(float(r["cup_x_m"]) for r in trials),
                "max_actual_x_m": max(float(r["cup_x_m"]) for r in trials),
                "min_actual_y_m": min(float(r["cup_y_m"]) for r in trials),
                "max_actual_y_m": max(float(r["cup_y_m"]) for r in trials),
            }
        )
    successes = sum(bool(r["success"]) for r in rows)
    summary = {
        "protocol": {
            "checkpoint": "030000",
            "execution_steps": rows[0]["execution_steps"] if rows else None,
            "rollout_seconds": rows[0]["rollout_seconds"] if rows else None,
            "cup_jitter_mm": rows[0]["cup_jitter_mm"] if rows else None,
            "random_seed": rows[0]["random_seed"] if rows else None,
            "success_criterion": (
                "cup lowest collision point >= 10 mm above table, no cup/table "
                "contact, continuously for >= 1.0 s"
            ),
            "random_perturbations": bool(rows and rows[0]["cup_jitter_mm"] > 0),
        },
        "completed_trials": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "positions": position_rows,
    }
    return position_rows, summary


def save_results(output_dir: Path, rows: list[dict]) -> None:
    rows.sort(key=lambda r: (int(r["position_index"]), int(r["trial_index"])))
    write_csv(output_dir / "trials.csv", rows)
    position_rows, summary = aggregate(rows)
    write_csv(output_dir / "summary_by_position.csv", position_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def result_from_diagnostics(
    diagnostics_path: Path,
    position_index: int,
    trial_index: int,
    nominal_cup_x: float,
    nominal_cup_y: float,
    cup_x: float,
    cup_y: float,
    offset_x_mm: float,
    offset_y_mm: float,
    cup_jitter_mm: float,
    seed: int,
    execution_steps: int,
    rollout_seconds: float,
    wall_seconds: float,
) -> dict:
    metrics = read_diagnostics(diagnostics_path)
    with diagnostics_path.open(newline="", encoding="utf-8") as handle:
        final_row = list(csv.DictReader(handle))[-1]
    return {
        "position_index": position_index,
        "trial_index": trial_index,
        "random_seed": seed,
        "cup_jitter_mm": cup_jitter_mm,
        "nominal_cup_x_m": nominal_cup_x,
        "nominal_cup_y_m": nominal_cup_y,
        "offset_x_mm": offset_x_mm,
        "offset_y_mm": offset_y_mm,
        "cup_x_m": cup_x,
        "cup_y_m": cup_y,
        "execution_steps": execution_steps,
        "rollout_seconds": rollout_seconds,
        **metrics,
        "final_cup_rise_mm": float(final_row["cup_center_rise_mm"]),
        "wall_seconds": wall_seconds,
        "diagnostics_csv": os.path.relpath(diagnostics_path, PROJECT_DIR),
    }


def main() -> None:
    args = parse_args()
    policy = args.policy.expanduser().resolve()
    default_output = DEFAULT_JITTER_OUTPUT if args.cup_jitter_mm > 0 else DEFAULT_OUTPUT
    output_dir = (args.output_dir or default_output).expanduser().resolve()
    selected = sorted(set(args.position_indices or range(len(POSITIONS))))
    if args.preview:
        print("position_index,trial_index,offset_x_mm,offset_y_mm,cup_x_m,cup_y_m")
        for position_index in selected:
            nominal_x, nominal_y = POSITIONS[position_index]
            for trial_index in range(1, args.repeats + 1):
                cup_x, cup_y, offset_x_mm, offset_y_mm = trial_position(
                    nominal_x,
                    nominal_y,
                    args.cup_jitter_mm,
                    args.seed,
                    position_index,
                    trial_index,
                )
                print(
                    f"{position_index},{trial_index},{offset_x_mm:.3f},"
                    f"{offset_y_mm:.3f},{cup_x:.6f},{cup_y:.6f}"
                )
        return
    if not policy.is_dir():
        raise FileNotFoundError(policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "trials.csv"
    if args.fresh and results_path.exists():
        raise FileExistsError(
            f"{results_path} already exists; choose a new --output-dir or omit --fresh"
        )
    rows = load_existing_results(
        results_path,
        args.execution_steps,
        args.rollout_seconds,
        args.cup_jitter_mm,
        args.seed,
    )
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "egl"

    total = len(selected) * args.repeats
    completed = 0
    for position_index in selected:
        nominal_cup_x, nominal_cup_y = POSITIONS[position_index]
        for trial_index in range(1, args.repeats + 1):
            completed += 1
            cup_x, cup_y, offset_x_mm, offset_y_mm = trial_position(
                nominal_cup_x,
                nominal_cup_y,
                args.cup_jitter_mm,
                args.seed,
                position_index,
                trial_index,
            )
            trial_dir = output_dir / f"position_{position_index:02d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            stem = trial_dir / f"trial_{trial_index:02d}"
            diagnostics_path = stem.with_suffix(".diagnostics.csv")
            log_path = stem.with_suffix(".log")

            if diagnostics_path.exists() and not args.fresh:
                result = result_from_diagnostics(
                    diagnostics_path,
                    position_index,
                    trial_index,
                    nominal_cup_x,
                    nominal_cup_y,
                    cup_x,
                    cup_y,
                    offset_x_mm,
                    offset_y_mm,
                    args.cup_jitter_mm,
                    args.seed,
                    args.execution_steps,
                    args.rollout_seconds,
                    0.0,
                )
                upsert(rows, result)
                save_results(output_dir, rows)
                print(
                    f"[{completed}/{total}] reused position {position_index}, "
                    f"trial {trial_index}: {'PASS' if result['success'] else 'FAIL'}",
                    flush=True,
                )
                continue
            if diagnostics_path.exists() and args.fresh:
                raise FileExistsError(
                    f"{diagnostics_path} already exists; choose a new --output-dir "
                    "or omit --fresh to resume"
                )

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
            ]
            if not args.save_videos:
                command.append("--act-no-video")

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
            wall_seconds = time.perf_counter() - started
            if process.returncode != 0:
                raise RuntimeError(
                    f"Position {position_index}, trial {trial_index} exited with "
                    f"code {process.returncode}; see {log_path}"
                )
            result = result_from_diagnostics(
                diagnostics_path,
                position_index,
                trial_index,
                nominal_cup_x,
                nominal_cup_y,
                cup_x,
                cup_y,
                offset_x_mm,
                offset_y_mm,
                args.cup_jitter_mm,
                args.seed,
                args.execution_steps,
                args.rollout_seconds,
                wall_seconds,
            )
            upsert(rows, result)
            save_results(output_dir, rows)
            print(
                f"[{completed}/{total}] position {position_index} "
                f"nominal=({nominal_cup_x:.3f}, {nominal_cup_y:.3f}), "
                f"offset=({offset_x_mm:+.2f}, {offset_y_mm:+.2f}) mm, "
                f"trial {trial_index}: "
                f"{'PASS' if result['success'] else 'FAIL'}, "
                f"time={result['duration_s']:.2f}s, "
                f"rise={result['final_cup_rise_mm']:.2f}mm",
                flush=True,
            )

    _, summary = aggregate(rows)
    print(
        f"Repeated evaluation complete: {summary['successes']}/"
        f"{summary['completed_trials']} PASS"
    )
    print(f"Per-trial results: {output_dir / 'trials.csv'}")
    print(f"Per-position summary: {output_dir / 'summary_by_position.csv'}")


if __name__ == "__main__":
    main()
