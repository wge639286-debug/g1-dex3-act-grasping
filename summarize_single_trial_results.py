#!/usr/bin/env python3
"""Combine the completed ten-position ACT single-trial benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

from evaluate_act_fixed_positions import read_diagnostics


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "project_results"

# Six trials were produced by evaluate_act_fixed_positions.py.  The final four
# were run interactively and are mapped here to their fixed benchmark positions.
MANUAL_TRIALS = [
    (6, 0.370, -0.025, PROJECT_DIR / "act_position_370_-025.diagnostics.csv"),
    (7, 0.400, -0.075, PROJECT_DIR / "act_position_400_-075.diagnostics.csv"),
    (8, 0.400, -0.050, PROJECT_DIR / "act_position_400_-050_30s.diagnostics.csv"),
    (9, 0.400, -0.025, PROJECT_DIR / "act_position_400_-025_30s.diagnostics.csv"),
]


def load_final_rise(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        return float(list(csv.DictReader(handle))[-1]["cup_center_rise_mm"])


def main() -> None:
    source = PROJECT_DIR / "act_fixed_position_eval" / "results.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    combined: list[dict] = []
    for row in rows:
        diagnostics_path = (
            PROJECT_DIR
            / "act_fixed_position_eval"
            / f"position_{int(row['position_index']):02d}.diagnostics.csv"
        )
        combined.append(
            {
                "position_index": int(row["position_index"]),
                "cup_x_m": float(row["cup_x_m"]),
                "cup_y_m": float(row["cup_y_m"]),
                "success": row["success"] == "True",
                "success_time_s": float(row["duration_s"]),
                "final_cup_rise_mm": load_final_rise(diagnostics_path),
                "max_lowest_clearance_mm": float(row["max_lowest_clearance_mm"]),
                "source": str(diagnostics_path.relative_to(PROJECT_DIR)),
            }
        )
    for position_index, cup_x, cup_y, path in MANUAL_TRIALS:
        metrics = read_diagnostics(path)
        combined.append(
            {
                "position_index": position_index,
                "cup_x_m": cup_x,
                "cup_y_m": cup_y,
                "success": metrics["success"],
                "success_time_s": metrics["duration_s"],
                "final_cup_rise_mm": load_final_rise(path),
                "max_lowest_clearance_mm": metrics["max_lowest_clearance_mm"],
                "source": path.name,
            }
        )
    combined.sort(key=lambda row: row["position_index"])
    if len(combined) != 10:
        raise ValueError(f"Expected 10 benchmark rows, found {len(combined)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "fixed_position_single_trial.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(combined[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(combined)

    successes = sum(row["success"] for row in combined)
    markdown = [
        "# ACT fixed-position benchmark (single trial)",
        "",
        "> These are single trials. They establish functional coverage, not a statistical success rate.",
        "",
        "| Index | Cup X (m) | Cup Y (m) | Result | Success time (s) | Final rise (mm) |",
        "|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in combined:
        rise_text = f"{row['final_cup_rise_mm']:.2f}"
        markdown.append(
            f"| {row['position_index']} | {row['cup_x_m']:.3f} | "
            f"{row['cup_y_m']:.3f} | {'PASS' if row['success'] else 'FAIL'} | "
            f"{row['success_time_s']:.2f} | {rise_text} |"
        )
    markdown += [
        "",
        f"Result: **{successes}/{len(combined)} positions passed once**.",
        "",
        "Protocol: checkpoint 030000, 25 action steps per replan, 30 s maximum. "
        "Success requires the cup's lowest collision point to remain at least 10 mm "
        "above the table, with no cup/table contact, for 1.0 continuous second.",
        "",
    ]
    md_path = OUTPUT_DIR / "fixed_position_single_trial.md"
    md_path.write_text("\n".join(markdown), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Single-trial benchmark: {successes}/{len(combined)} PASS")


if __name__ == "__main__":
    main()
