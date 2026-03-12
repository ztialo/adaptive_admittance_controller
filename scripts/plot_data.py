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
    parser.add_argument("--title", type=str, default=None, help="Optional figure title override")
    parser.add_argument("--show", action="store_true", help="Display plot window")
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    steps: list[int] = []
    times: list[datetime] = []
    force_values: list[float] = []
    f_des_values: list[float] = []
    has_z_cols = False
    x_curr_values: list[float] = []
    x_nom_values: list[float] = []
    x_cmd_values: list[float] = []
    x_dot_values: list[float] = []
    x_ddot_values: list[float] = []
    stiffness_values: list[float] = []
    damping_values: list[float] = []
    youngs_modulus_values: list[float] = []
    mode_values: list[str] = []
    x_cmd_n_values: list[float] = []
    x_curr_n_values: list[float] = []
    tracking_error_n_values: list[float] = []
    admittance_integrate_enabled_values: list[float] = []
    non_contact_correction_mag_values: list[float] = []
    ik_target_delta_norm_values: list[float] = []
    ik_target_out_of_limits_values: list[float] = []
    x_cmd_step_clipped_values: list[float] = []

    with args.csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        fieldnames = set(reader.fieldnames)
        has_z_cols = "ee_pos_b_z" in fieldnames and "ee_goal_pos_b_z" in fieldnames
        has_old_stiffness_col = "wall_youngs_modulus_pa" in fieldnames
        has_new_stiffness_col = "wall_compliant_contact_stiffness" in fieldnames
        has_new_damping_col = "wall_compliant_contact_damping" in fieldnames
        has_youngs_modulus_col = "youngs_modulus_pa" in fieldnames
        has_mode_col = "mode" in fieldnames
        has_x_cmd_col = "x_cmd_b_z" in fieldnames
        has_x_dot_col = "admittance_velocity" in fieldnames
        has_x_ddot_col = "admittance_acceleration" in fieldnames
        has_x_cmd_n_col = "x_cmd_n" in fieldnames
        has_x_curr_n_col = "x_curr_n" in fieldnames
        has_tracking_error_n_col = "tracking_error_n" in fieldnames
        has_aw_gate_col = "admittance_integrate_enabled" in fieldnames
        has_nc_mag_col = "non_contact_correction_mag" in fieldnames
        has_ik_delta_col = "ik_target_delta_norm" in fieldnames
        has_ik_oob_col = "ik_target_out_of_limits" in fieldnames
        has_step_clip_col = "x_cmd_step_clipped" in fieldnames

        has_f_des_col = "f_des_n" in fieldnames
        has_f_comp_filt_col = "f_compression_pos_filt" in fieldnames
        has_fz_col = "fz" in fieldnames
        force_col_name = "f_compression_pos_filt" if has_f_comp_filt_col else "fz"
        required = {"wall_time_iso", "step", force_col_name}
        if not required.issubset(fieldnames):
            missing = sorted(required - fieldnames)
            raise ValueError(f"CSV missing required columns: {missing}")

        for row in reader:
            steps.append(int(row["step"]))
            times.append(datetime.fromisoformat(row["wall_time_iso"]))
            force_values.append(_float_or_nan(row.get(force_col_name)))
            if has_f_des_col:
                f_des_values.append(_float_or_nan(row.get("f_des_n")))

            if has_z_cols:
                x_curr_values.append(_float_or_nan(row.get("ee_pos_b_z")))
                x_nom_values.append(_float_or_nan(row.get("ee_goal_pos_b_z")))
            if has_x_cmd_col:
                x_cmd_values.append(_float_or_nan(row.get("x_cmd_b_z")))
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
            if has_mode_col:
                mode_values.append((row.get("mode") or "").strip())
            if has_x_cmd_n_col:
                x_cmd_n_values.append(_float_or_nan(row.get("x_cmd_n")))
            if has_x_curr_n_col:
                x_curr_n_values.append(_float_or_nan(row.get("x_curr_n")))
            if has_tracking_error_n_col:
                tracking_error_n_values.append(_float_or_nan(row.get("tracking_error_n")))
            if has_aw_gate_col:
                admittance_integrate_enabled_values.append(_float_or_nan(row.get("admittance_integrate_enabled")))
            if has_nc_mag_col:
                non_contact_correction_mag_values.append(_float_or_nan(row.get("non_contact_correction_mag")))
            if has_ik_delta_col:
                ik_target_delta_norm_values.append(_float_or_nan(row.get("ik_target_delta_norm")))
            if has_ik_oob_col:
                ik_target_out_of_limits_values.append(_float_or_nan(row.get("ik_target_out_of_limits")))
            if has_step_clip_col:
                x_cmd_step_clipped_values.append(_float_or_nan(row.get("x_cmd_step_clipped")))

    if not force_values:
        raise ValueError("No valid rows found in CSV.")

    force_meas = np.asarray(force_values, dtype=float)
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

    force_label = "Fcomp" if force_col_name == "f_compression_pos_filt" else "Fz"
    force_ax.plot(x, force_meas, linewidth=1.4, color="tab:blue", label=force_label)
    if f_des is not None and len(f_des) == len(force_meas):
        force_ax.plot(x, f_des, linewidth=1.4, color="tab:orange", label="Fdes")
    force_ax.set_title("Contact Force")
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
    if mode_values:
        unique_modes = sorted(set([m for m in mode_values if m]))
        if len(unique_modes) == 1:
            subtitle = f"Mode: {unique_modes[0]}"
    if youngs_modulus_values:
        ym = np.asarray(youngs_modulus_values, dtype=float)
        unique_ym = np.unique(ym[np.isfinite(ym)])
        if len(unique_ym) == 1:
            subtitle += f" | Young's modulus: {unique_ym[0]:.3g} Pa"
    if args.title is not None:
        title = f"{args.title}\n{subtitle}" if subtitle else args.title
    else:
        title = f"{run_family_name}\n{subtitle}"
    if stiffness_values:
        stiff = np.asarray(stiffness_values, dtype=float)
        unique_stiff = np.unique(stiff[np.isfinite(stiff)])
        if len(unique_stiff) == 1:
            if damping_values:
                title += f" | compliant k={unique_stiff[0]:.3g}"
            else:
                title += f" | Young's modulus: {unique_stiff[0]:.3g} Pa"
    if damping_values:
        damp = np.asarray(damping_values, dtype=float)
        unique_damp = np.unique(damp[np.isfinite(damp)])
        if len(unique_damp) == 1:
            title += f", d={unique_damp[0]:.3g}"

    if has_z_cols:
        x_curr = np.asarray(x_curr_values, dtype=float)
        x_nom = np.asarray(x_nom_values, dtype=float)
        x_pos_ax.plot(x, x_nom, color="tab:gray", linewidth=1.2, label="z_nom (no compression)")
        x_pos_ax.plot(x, x_curr, color="tab:green", linewidth=1.2, label="z_curr (compression)")
        if x_cmd_values and len(x_cmd_values) == len(x):
            x_cmd = np.asarray(x_cmd_values, dtype=float)
            x_pos_ax.plot(x, x_cmd, color="tab:blue", linewidth=1.2, label="z_cmd (admittance)")
        x_pos_ax.set_title("Z Position: Nominal vs Current")
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
        x_pos_ax.set_title("Z Position: unavailable in CSV")
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

    # Optional diagnostics figure for command-path / windup analysis.
    has_diag = (
        len(x_cmd_n_values) == len(x)
        and len(x_curr_n_values) == len(x)
        and len(tracking_error_n_values) == len(x)
    )
    if has_diag:
        fig2, ax2 = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
        ax_force, ax_adm, ax_xn, ax_track, ax_pipe = ax2

        ax_force.plot(x, force_meas, linewidth=1.2, color="tab:blue", label=force_label)
        if f_des is not None and len(f_des) == len(force_meas):
            ax_force.plot(x, f_des, linewidth=1.2, color="tab:orange", label="F_des")
            ax_force.plot(x, f_des - force_meas, linewidth=1.1, color="tab:red", label="F_err")
        ax_force.set_ylabel("N")
        ax_force.set_title("Force")
        ax_force.grid(True, alpha=0.3)
        ax_force.legend(loc="best")

        if x_dot_values and x_ddot_values and len(x_dot_values) == len(x) and len(x_ddot_values) == len(x):
            ax_adm.plot(x, np.asarray(x_dot_values), color="tab:purple", linewidth=1.1, label="x_dot")
            ax_adm.plot(x, np.asarray(x_ddot_values), color="tab:green", linewidth=1.1, label="x_ddot")
        ax_adm.plot(x, np.asarray(x_cmd_n_values), color="tab:gray", linewidth=1.1, label="x_cmd_n")
        ax_adm.set_ylabel("m, m/s, m/s^2")
        ax_adm.set_title("Admittance States")
        ax_adm.grid(True, alpha=0.3)
        ax_adm.legend(loc="best")

        ax_xn.plot(x, np.asarray(x_cmd_n_values), color="tab:blue", linewidth=1.2, label="x_cmd_n")
        ax_xn.plot(x, np.asarray(x_curr_n_values), color="tab:green", linewidth=1.2, label="x_curr_n")
        ax_xn.set_ylabel("m")
        ax_xn.set_title("Normal-Axis Command Tracking")
        ax_xn.grid(True, alpha=0.3)
        ax_xn.legend(loc="best")

        ax_track.plot(x, np.asarray(tracking_error_n_values), color="tab:red", linewidth=1.2, label="tracking_error_n")
        if len(admittance_integrate_enabled_values) == len(x):
            ax_track.plot(
                x,
                np.asarray(admittance_integrate_enabled_values),
                color="tab:olive",
                linewidth=1.0,
                label="integrate_enabled",
            )
        ax_track.set_ylabel("m / bool")
        ax_track.set_title("Tracking Error + Anti-Windup Gate")
        ax_track.grid(True, alpha=0.3)
        ax_track.legend(loc="best")

        if len(non_contact_correction_mag_values) == len(x):
            ax_pipe.plot(
                x, np.asarray(non_contact_correction_mag_values), color="tab:brown", linewidth=1.2, label="non_contact_corr_mag"
            )
        if len(ik_target_delta_norm_values) == len(x):
            ax_pipe.plot(x, np.asarray(ik_target_delta_norm_values), color="tab:cyan", linewidth=1.1, label="ik_target_delta_norm")
        if len(ik_target_out_of_limits_values) == len(x):
            ax_pipe.plot(x, np.asarray(ik_target_out_of_limits_values), color="tab:pink", linewidth=1.0, label="ik_target_out_of_limits")
        if len(x_cmd_step_clipped_values) == len(x):
            ax_pipe.plot(x, np.asarray(x_cmd_step_clipped_values), color="tab:gray", linewidth=1.0, label="x_cmd_step_clipped")
        ax_pipe.set_ylabel("mixed")
        ax_pipe.set_xlabel(x_label)
        ax_pipe.set_title("Pipeline Indicators")
        ax_pipe.grid(True, alpha=0.3)
        if ax_pipe.has_data():
            ax_pipe.legend(loc="best")

        fig2.suptitle(f"{title}\nDiagnostics")
        plt.tight_layout()
        diag_save_path = save_path.with_name(f"{save_path.stem}_diag{save_path.suffix}")
        fig2.savefig(diag_save_path, dpi=160)
        print(f"Saved diagnostics plot to: {diag_save_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig2)


if __name__ == "__main__":
    main()
