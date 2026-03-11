#!/usr/bin/env python3
"""Plot admittance/baseline CSV logs."""

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
    parser = argparse.ArgumentParser(description="Plot admittance/baseline CSV logs")
    parser.add_argument("csv_path", type=Path, help="Path to CSV file")
    parser.add_argument("--x-axis", choices=["step", "time"], default="step", help="X-axis type")
    parser.add_argument("--save", type=Path, default=None, help="Output image path (default: CSV stem + .png)")
    parser.add_argument("--show", action="store_true", help="Display plot window")
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    steps: list[int] = []
    times: list[datetime] = []
    fz_values: list[float] = []
    f_des_values: list[float] = []
    has_x_cols = False
    x_curr_values: list[float] = []
    x_nom_values: list[float] = []
    x_cmd_values: list[float] = []
    x_dot_values: list[float] = []
    x_ddot_values: list[float] = []
    stiffness_values: list[float] = []
    damping_values: list[float] = []
    youngs_modulus_values: list[float] = []

    with args.csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        fieldnames = set(reader.fieldnames)
        has_x_cols = "ee_pos_b_x" in fieldnames and "ee_goal_pos_b_x" in fieldnames
        has_old_stiffness_col = "wall_youngs_modulus_pa" in fieldnames
        has_new_stiffness_col = "wall_compliant_contact_stiffness" in fieldnames
        has_new_damping_col = "wall_compliant_contact_damping" in fieldnames
        has_youngs_modulus_col = "youngs_modulus_pa" in fieldnames
        has_x_cmd_col = "x_cmd_b_x" in fieldnames
        has_x_dot_col = "admittance_velocity" in fieldnames
        has_x_ddot_col = "admittance_acceleration" in fieldnames

        has_f_des_col = "f_des_n" in fieldnames
        required = {"wall_time_iso", "step", "fz"}
        if not required.issubset(fieldnames):
            missing = sorted(required - fieldnames)
            raise ValueError(f"CSV missing required columns: {missing}")

        for row in reader:
            steps.append(int(row["step"]))
            times.append(datetime.fromisoformat(row["wall_time_iso"]))
            fz_values.append(_float_or_nan(row.get("fz")))
            if has_f_des_col:
                f_des_values.append(_float_or_nan(row.get("f_des_n")))

            if has_x_cols:
                x_curr_values.append(_float_or_nan(row.get("ee_pos_b_x")))
                x_nom_values.append(_float_or_nan(row.get("ee_goal_pos_b_x")))
            if has_x_cmd_col:
                x_cmd_values.append(_float_or_nan(row.get("x_cmd_b_x")))
            if has_x_dot_col:
                x_dot_values.append(_float_or_nan(row.get("admittance_velocity")))
            if has_x_ddot_col:
                x_ddot_values.append(_float_or_nan(row.get("admittance_acceleration")))
            if has_old_stiffness_col:
                stiffness_values.append(_float_or_nan(row.get("wall_youngs_modulus_pa")))
            if has_new_stiffness_col:
                stiffness_values.append(_float_or_nan(row.get("wall_compliant_contact_stiffness")))
            if has_new_damping_col:
                damping_values.append(_float_or_nan(row.get("wall_compliant_contact_damping")))
            if has_youngs_modulus_col:
                youngs_modulus_values.append(_float_or_nan(row.get("youngs_modulus_pa")))

    if not fz_values:
        raise ValueError("No valid rows found in CSV.")

    fz = np.asarray(fz_values, dtype=float)
    f_des = np.asarray(f_des_values, dtype=float) if f_des_values else None
    if args.x_axis == "time":
        t0 = times[0]
        x = np.array([(t - t0).total_seconds() for t in times], dtype=float)
        x_label = "Time (s)"
    else:
        x = np.asarray(steps, dtype=float)
        x_label = "Step"

    # Layout: always 3 rows x 1 column so x-axis lines up across plots.
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    force_ax, x_pos_ax, x_dyn_ax = axes

    force_ax.plot(x, fz, linewidth=1.4, color="tab:blue", label="Fz")
    if f_des is not None and len(f_des) == len(fz):
        force_ax.plot(x, f_des, linewidth=1.4, color="tab:orange", label="Fdes")
    force_ax.set_title("Contact Force (Fz)")
    force_ax.set_xlabel(x_label)
    force_ax.set_ylabel("N")
    force_ax.grid(True, alpha=0.3)
    force_ax.legend(loc="best")

    # Prefer the log family folder name (e.g. ".../admittance_baseline/<run>/ft_env0.csv").
    if args.csv_path.parent.parent.exists():
        run_family_name = args.csv_path.parent.parent.name
    else:
        run_family_name = args.csv_path.parent.name
    subtitle = args.csv_path.name
    if youngs_modulus_values:
        unique_ym = np.unique(np.asarray(youngs_modulus_values, dtype=float))
        if len(unique_ym) == 1:
            subtitle = f"Young's modulus: {unique_ym[0]:.3g} Pa"
    title = f"{run_family_name}\n{subtitle}"
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

    if has_x_cols:
        x_curr = np.asarray(x_curr_values, dtype=float)
        x_nom = np.asarray(x_nom_values, dtype=float)
        x_pos_ax.plot(x, x_nom, color="tab:gray", linewidth=1.2, label="x_nom (no compression)")
        x_pos_ax.plot(x, x_curr, color="tab:green", linewidth=1.2, label="x_curr (compression)")
        if x_cmd_values and len(x_cmd_values) == len(x):
            x_cmd = np.asarray(x_cmd_values, dtype=float)
            x_pos_ax.plot(x, x_cmd, color="tab:blue", linewidth=1.2, label="x_cmd (admittance)")
        x_pos_ax.set_title("X Position: Nominal vs Current")
        x_pos_ax.set_ylabel("m")
        x_pos_ax.grid(True, alpha=0.3)
        x_pos_ax.legend(loc="best")

        if x_dot_values and x_ddot_values and len(x_dot_values) == len(x) and len(x_ddot_values) == len(x):
            x_dot = np.asarray(x_dot_values, dtype=float)
            x_ddot = np.asarray(x_ddot_values, dtype=float)
            x_dyn_ax.plot(x, x_dot, color="tab:purple", linewidth=1.2, label="x_dot")
            x_dyn_ax.plot(x, x_ddot, color="tab:red", linewidth=1.2, label="x_ddot")
            x_dyn_ax.set_title("Admittance Dynamics")
        else:
            x_dyn_ax.set_title("Admittance Dynamics: unavailable in CSV")
    else:
        x_pos_ax.set_title("X Position: unavailable in CSV")
        x_pos_ax.set_ylabel("m")
        x_pos_ax.grid(True, alpha=0.3)
        x_dyn_ax.set_title("Admittance Dynamics: unavailable in CSV")

    x_dyn_ax.set_xlabel(x_label)
    x_dyn_ax.set_ylabel("m/s, m/s^2")
    x_dyn_ax.grid(True, alpha=0.3)
    if x_dyn_ax.has_data():
        x_dyn_ax.legend(loc="best")

    fig.suptitle(title)
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
