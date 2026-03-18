#!/usr/bin/env python3
"""Plot admittance/baseline CSV logs."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


K_ENV_PLOT_FORCE_THRESHOLD_N = 2.0
K_ENV_PLOT_CONFIRM_STEPS = 3
K_ENV_PLOT_RESET_STEPS = 500


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

    # Optional diagnostics figure with environment-stiffness estimation.
    has_diag = len(x_cmd_n_values) == len(x) and len(x_curr_n_values) == len(x)
    if has_diag:
        fig2, ax2 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        ax_force, ax_adm, ax_xn, ax_kenv = ax2

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

        x_curr_n = np.asarray(x_curr_n_values, dtype=float)
        k_env_active = np.zeros_like(x_curr_n, dtype=bool)
        confirm_counter = 0
        for idx, force_val in enumerate(force_meas):
            if idx % K_ENV_PLOT_RESET_STEPS == 0:
                confirm_counter = 0
            if np.isfinite(force_val) and force_val > K_ENV_PLOT_FORCE_THRESHOLD_N:
                confirm_counter += 1
            else:
                confirm_counter = 0
            if confirm_counter >= K_ENV_PLOT_CONFIRM_STEPS:
                k_env_active[idx] = True
        for start_idx in range(0, len(k_env_active), K_ENV_PLOT_RESET_STEPS):
            end_idx = min(start_idx + K_ENV_PLOT_RESET_STEPS, len(k_env_active))
            k_env_active[start_idx:end_idx] = np.logical_or.accumulate(k_env_active[start_idx:end_idx])
        min_penetration = 1.0e-5
        k_env_est = np.zeros_like(x_curr_n, dtype=float)
        valid_k_env = np.isfinite(force_meas) & np.isfinite(x_curr_n) & (x_curr_n > min_penetration) & k_env_active
        k_env_est[valid_k_env] = force_meas[valid_k_env] / x_curr_n[valid_k_env]

        k_env_est_smooth = np.zeros_like(k_env_est, dtype=float)
        alpha = 0.15
        ema_val = 0.0
        for idx, value in enumerate(k_env_est):
            if k_env_active[idx] and np.isfinite(value):
                ema_val = alpha * value + (1.0 - alpha) * ema_val
            else:
                ema_val = 0.0
            k_env_est_smooth[idx] = ema_val

        if np.any(np.isfinite(k_env_est_smooth)):
            ax_kenv.plot(x, k_env_est_smooth, color="tab:red", linewidth=1.4, label="K_env est EMA")
        if stiffness_values:
            stiff = np.asarray(stiffness_values, dtype=float)
            finite_stiff = stiff[np.isfinite(stiff)]
            if finite_stiff.size > 0:
                unique_stiff = np.unique(finite_stiff)
                if unique_stiff.size == 1:
                    ax_kenv.axhline(unique_stiff[0], color="tab:blue", linestyle="--", linewidth=1.0, label="configured stiffness")
        ax_kenv.set_ylabel("N/m")
        ax_kenv.set_xlabel(x_label)
        ax_kenv.set_title("Environment Stiffness Estimate")
        ax_kenv.grid(True, alpha=0.3)
        if ax_kenv.has_data():
            ax_kenv.legend(loc="best")

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
