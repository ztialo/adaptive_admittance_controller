#!/usr/bin/env python3
"""Plot baseline_osc FT/EE logs (supports old and new CSV schemas)."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _float_or_nan(value: str | None) -> float:
    if value is None or value == "":
        return np.nan
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Plot baseline_osc CSV logs")
    parser.add_argument("csv_path", type=Path, help="Path to CSV file")
    parser.add_argument("--x-axis", choices=["step", "time"], default="step", help="X-axis type")
    parser.add_argument("--save", type=Path, default=None, help="Output image path (default: CSV stem + .png)")
    parser.add_argument("--show", action="store_true", help="Display plot window")
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    steps: list[int] = []
    times: list[datetime] = []
    wrench_rows: list[list[float]] = []
    has_ee_cols = False
    ee_err_norm: list[float] = []
    ee_pos_err_xyz: list[list[float]] = []
    stiffness_values: list[float] = []
    damping_values: list[float] = []

    with args.csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        fieldnames = set(reader.fieldnames)
        has_ee_cols = "ee_pos_err_b_norm" in fieldnames
        has_old_stiffness_col = "wall_youngs_modulus_pa" in fieldnames
        has_new_stiffness_col = "wall_compliant_contact_stiffness" in fieldnames
        has_new_damping_col = "wall_compliant_contact_damping" in fieldnames

        required = {"wall_time_iso", "step", "fx", "fy", "fz", "tx", "ty", "tz"}
        if not required.issubset(fieldnames):
            missing = sorted(required - fieldnames)
            raise ValueError(f"CSV missing required columns: {missing}")

        for row in reader:
            steps.append(int(row["step"]))
            times.append(datetime.fromisoformat(row["wall_time_iso"]))
            wrench_rows.append(
                [
                    _float_or_nan(row.get("fx")),
                    _float_or_nan(row.get("fy")),
                    _float_or_nan(row.get("fz")),
                    _float_or_nan(row.get("tx")),
                    _float_or_nan(row.get("ty")),
                    _float_or_nan(row.get("tz")),
                ]
            )

            if has_ee_cols:
                ee_err_norm.append(_float_or_nan(row.get("ee_pos_err_b_norm")))
                ee_pos_err_xyz.append(
                    [
                        _float_or_nan(row.get("ee_pos_err_b_x")),
                        _float_or_nan(row.get("ee_pos_err_b_y")),
                        _float_or_nan(row.get("ee_pos_err_b_z")),
                    ]
                )
            if has_old_stiffness_col:
                stiffness_values.append(_float_or_nan(row.get("wall_youngs_modulus_pa")))
            if has_new_stiffness_col:
                stiffness_values.append(_float_or_nan(row.get("wall_compliant_contact_stiffness")))
            if has_new_damping_col:
                damping_values.append(_float_or_nan(row.get("wall_compliant_contact_damping")))

    if not wrench_rows:
        raise ValueError("No valid rows found in CSV.")

    wrench = np.asarray(wrench_rows, dtype=float)
    if args.x_axis == "time":
        t0 = times[0]
        x = np.array([(t - t0).total_seconds() for t in times], dtype=float)
        x_label = "Time (s)"
    else:
        x = np.asarray(steps, dtype=float)
        x_label = "Step"

    # Layout: top 2x3 = wrench always; bottom row for EE error if present.
    if has_ee_cols:
        fig = plt.figure(figsize=(14, 11))
        gs = fig.add_gridspec(3, 3)
        wrench_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
        err_norm_ax = fig.add_subplot(gs[2, 0])
        err_xyz_ax = fig.add_subplot(gs[2, 1:])
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
        wrench_axes = list(axes.flat)

    labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    for i, ax in enumerate(wrench_axes):
        ax.plot(x, wrench[:, i], linewidth=1.2, color="tab:blue")
        ax.set_title(labels[i])
        ax.grid(True, alpha=0.3)
        if i >= 3:
            ax.set_xlabel(x_label)

    title = f"Baseline OSC Log\n{args.csv_path.name}"
    if stiffness_values:
        unique_stiff = np.unique(np.asarray(stiffness_values, dtype=float))
        if len(unique_stiff) == 1:
            if damping_values:
                title += f" | compliant k={unique_stiff[0]:.3g}"
            else:
                title += f" | Young's modulus: {unique_stiff[0]:.3g} Pa"
    if damping_values:
        unique_damp = np.unique(np.asarray(damping_values, dtype=float))
        if len(unique_damp) == 1:
            title += f", d={unique_damp[0]:.3g}"
    fig.suptitle(title)

    if has_ee_cols:
        err_norm = np.asarray(ee_err_norm, dtype=float)
        err_xyz = np.asarray(ee_pos_err_xyz, dtype=float)
        err_norm_ax.plot(x, err_norm, color="tab:red", linewidth=1.2)
        err_norm_ax.set_title("EE Position Error Norm")
        err_norm_ax.set_xlabel(x_label)
        err_norm_ax.set_ylabel("m")
        err_norm_ax.grid(True, alpha=0.3)

        err_xyz_ax.plot(x, err_xyz[:, 0], label="err_x", linewidth=1.1)
        err_xyz_ax.plot(x, err_xyz[:, 1], label="err_y", linewidth=1.1)
        err_xyz_ax.plot(x, err_xyz[:, 2], label="err_z", linewidth=1.1)
        err_xyz_ax.set_title("EE Position Error Components")
        err_xyz_ax.set_xlabel(x_label)
        err_xyz_ax.set_ylabel("m")
        err_xyz_ax.grid(True, alpha=0.3)
        err_xyz_ax.legend(loc="best")

    plt.tight_layout()
    save_path = args.save if args.save is not None else args.csv_path.with_suffix(".png")
    fig.savefig(save_path, dpi=160)
    print(f"Saved plot to: {save_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
