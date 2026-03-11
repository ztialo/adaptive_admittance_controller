# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Baseline 1D admittance force-control experiment against a rigid wall.

Flow:
1) Move EE to pre-contact waypoint.
2) Move EE to final contact pose.
3) Latch contact using threshold + confirm steps.
4) Run 1D admittance along wall normal with force ramp + force LPF.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# CLI
parser = argparse.ArgumentParser(description="Rigid-wall baseline 1D admittance control.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (baseline uses 1).")
parser.add_argument("--log", action="store_true", default=False, help="Enable CSV/video logging.")
parser.add_argument("--record_length", type=int, default=0, help="If >0, stop after this many steps.")

# Required baseline knobs
parser.add_argument("--desired_contact_force", type=float, default=10.0, help="Desired normal force (N).")
parser.add_argument("--admittance_M", type=float, default=2.0, help="Admittance virtual mass.")
parser.add_argument("--admittance_B", type=float, default=220.0, help="Admittance virtual damping.")
parser.add_argument("--admittance_K", type=float, default=160.0, help="Admittance virtual stiffness.")
parser.add_argument("--contact_force_threshold", type=float, default=1.0, help="Contact threshold (N).")
parser.add_argument("--contact_confirm_steps", type=int, default=3, help="Consecutive threshold hits to latch contact.")
parser.add_argument("--force_ramp_time", type=float, default=0.5, help="Ramp time (s) from 0 to desired force.")
parser.add_argument("--force_filter_alpha", type=float, default=0.2, help="LPF alpha for measured force [0,1].")
parser.add_argument("--max_admittance_offset", type=float, default=0.05, help="Clamp for admittance offset (m).")
parser.add_argument("--max_admittance_velocity", type=float, default=0.05, help="Clamp for admittance velocity (m/s).")
parser.add_argument("--debug_print_every", type=int, default=20, help="Print every N steps (0 disables).")
parser.add_argument(
    "--max_final_approach_speed",
    type=float,
    default=0.01,
    help="Max Cartesian speed (m/s) for final pre-contact approach command.",
)

# Waypoint / robustness helpers
parser.add_argument("--contact_detection_delay_steps", type=int, default=20, help="Delay before contact detection after reset.")
parser.add_argument("--waypoint_offset", type=float, default=0.015, help="Waypoint retreat along -contact-axis (m).")
parser.add_argument("--wall_penetration_depth", type=float, default=0.075, help="Final goal penetration depth into wall along contact axis (m).")
parser.add_argument("--waypoint_switch_pos_thresh", type=float, default=0.008, help="Waypoint switch threshold (m).")
parser.add_argument("--waypoint_hold_steps", type=int, default=15, help="Consecutive waypoint-hold steps.")
parser.add_argument("--enable_waypoint", action="store_true", default=True, help="Enable pre-contact waypoint stage.")
parser.add_argument("--non_contact_hold_gain", type=float, default=2.0, help="Gain for non-contact-axis hold.")
parser.add_argument(
    "--max_non_contact_correction", type=float, default=0.01, help="Per-axis clamp for non-contact correction (m)."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Baseline is intentionally single-env for clarity.
if args_cli.num_envs != 1:
    print(f"[WARN] Forcing num_envs=1 for baseline (requested {args_cli.num_envs}).")
    args_cli.num_envs = 1

# Clamp alpha into [0, 1]
args_cli.force_filter_alpha = max(0.0, min(1.0, args_cli.force_filter_alpha))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Delayed imports after app launch
# pylint: disable=wrong-import-position
import csv  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat, quat_apply, quat_apply_inverse, subtract_frame_transforms

from source.franka import FRANKA_3_HIGH_PD_CFG  # noqa: E402


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Rigid-wall baseline scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    rigid_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RigidWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.8, 0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0), opacity=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7, restitution=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.50, -0.3, 0.85), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    robot = FRANKA_3_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["franka_shoulder"].stiffness = 500.0
    robot.actuators["franka_shoulder"].damping = 80.0
    robot.actuators["franka_forearm"].stiffness = 500.0
    robot.actuators["franka_forearm"].damping = 80.0
    robot.spawn.rigid_props.disable_gravity = True

    observer_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ObserverCamera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )


def update_states(robot: Articulation, ee_frame_idx: int, arm_joint_ids: list[int]):
    """Update robot kinematics in root/body frame and world frame."""
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(robot.data.root_quat_w).transpose(-1, -2)
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]

    return jacobian_b, ee_pose_b, root_pose_w, ee_pose_w, joint_pos


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    observer_camera = scene["observer_camera"]

    ee_frame_name = "fr3_leftfinger"
    arm_joint_names = ["fr3_joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]
    env_origins = scene.env_origins

    # Observer camera + main camera matching viewpoint.
    camera_positions = env_origins + torch.tensor([-0.25, 1.8, 1.55], device=sim.device, dtype=env_origins.dtype)
    camera_targets = env_origins + torch.tensor([0.38, 0.0, 0.86], device=sim.device, dtype=env_origins.dtype)
    observer_camera.set_world_poses_from_view(camera_positions, camera_targets)

    # Force sensing body
    ft_body_name = "fr3_hand"
    ft_body_idx = None
    try:
        body_ids, _ = robot.find_bodies(ft_body_name)
        if len(body_ids) > 0:
            ft_body_idx = int(body_ids[0])
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Failed to resolve FT body index: {exc}")

    # Logging
    ft_csv_file = None
    ft_writer = None
    ft_log_path = None
    run_dir = None
    if args_cli.log:
        logs_root = REPO_ROOT / "logs" / "admittance_baseline_rigid"
        logs_root.mkdir(parents=True, exist_ok=True)
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = logs_root / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)
        ft_log_path = run_dir / "ft_env0.csv"
        ft_csv_file = open(ft_log_path, "w", newline="", encoding="utf-8")
        ft_writer = csv.writer(ft_csv_file)
        ft_writer.writerow(
            [
                "wall_time_iso",
                "step",
                "ee_pos_b_x",
                "ee_pos_b_y",
                "ee_pos_b_z",
                "ee_goal_pos_b_x",
                "ee_goal_pos_b_y",
                "ee_goal_pos_b_z",
                "x_cmd_b_x",
                "x_cmd_b_y",
                "x_cmd_b_z",
                "x_n",
                "fz",
                "f_ext_n_raw",
                "f_ext_n_filt",
                "f_des_n",
                "f_err_n",
                "admittance_offset",
                "admittance_velocity",
                "admittance_acceleration",
                "contact_active",
                "phase",
            ]
        )
        print(f"[INFO] FT log file: {ft_log_path}")

    video_writer = None
    video_path = None
    if args_cli.log and run_dir is not None:
        video_path = run_dir / "admittance_baseline_env0.mp4"
        fps = max(1, int(round(1.0 / sim.get_physics_dt())))
        video_writer = imageio.get_writer(video_path, fps=fps)

    # Differential IK
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Rigid wall geometry in root/body frame (matches SceneCfg)
    wall_center_b = torch.tensor([0.50, -0.3, 0.85], device=sim.device)
    wall_quat_b = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
    wall_thickness = 0.06
    wall_rot_mat_b = matrix_from_quat(wall_quat_b.unsqueeze(0)).squeeze(0)
    wall_x_axis_b = wall_rot_mat_b[:, 0]
    wall_y_axis_b = wall_rot_mat_b[:, 1]
    wall_z_axis_b = wall_rot_mat_b[:, 2]

    # Near-side face normal points from wall to robot (-X). Compression axis into wall is +X.
    wall_normal_out_b = -wall_x_axis_b
    contact_axis_b = -wall_normal_out_b
    wall_surface_center_b = wall_center_b + 0.5 * wall_thickness * wall_normal_out_b

    # Goal set (3 y-locations), fixed orientation.
    wall_plane_offsets_yz = torch.tensor([[0.10, 0.0], [0.0, 0.0], [-0.10, 0.0]], device=sim.device)
    wall_contact_center_b = wall_surface_center_b + args_cli.wall_penetration_depth * contact_axis_b
    ee_goal_pos_set_b = (
        wall_contact_center_b.unsqueeze(0)
        + wall_plane_offsets_yz[:, 0:1] * wall_y_axis_b.unsqueeze(0)
        + wall_plane_offsets_yz[:, 1:2] * wall_z_axis_b.unsqueeze(0)
    )
    # Baseline experiment requirement: keep nominal contact x target fixed.
    ee_goal_pos_set_b[:, 0] = 0.5138
    ee_goal_quat_set_b = torch.tensor([0.0, 0.70710678, 0.0, 0.70710678], device=sim.device).repeat(3, 1)
    ee_goal_pose_set_b = torch.cat([ee_goal_pos_set_b, ee_goal_quat_set_b], dim=-1)

    ee_waypoint_pos_set_b = ee_goal_pos_set_b - args_cli.waypoint_offset * contact_axis_b.unsqueeze(0)
    ee_waypoint_pose_set_b = torch.cat([ee_waypoint_pos_set_b, ee_goal_quat_set_b], dim=-1)

    sim_dt = sim.get_physics_dt()
    robot.update(sim_dt)

    jacobian_b, ee_pose_b, root_pose_w, ee_pose_w, joint_pos = update_states(robot, ee_frame_idx, arm_joint_ids)

    # Controller state
    current_goal_idx = 0
    active_goal_idx = 0
    moving_to_waypoint = True
    waypoint_hold_counter = 0

    contact_active = torch.zeros(scene.num_envs, dtype=torch.bool, device=sim.device)
    prev_contact_active = torch.zeros_like(contact_active)
    contact_confirm_counter = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    contact_latch_step = torch.full((scene.num_envs,), -1, dtype=torch.long, device=sim.device)

    force_filt_n = torch.zeros(scene.num_envs, device=sim.device)
    admittance_offset = torch.zeros(scene.num_envs, device=sim.device)
    admittance_velocity = torch.zeros(scene.num_envs, device=sim.device)
    admittance_acceleration = torch.zeros(scene.num_envs, device=sim.device)

    ik_commands = torch.zeros(scene.num_envs, diff_ik.action_dim, device=sim.device)
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)
    joint_pos_des = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)
    x_cmd_prev_b = ee_pose_b[:, 0:3].clone()

    steps_since_reset = 0
    count = 0

    try:
        while simulation_app.is_running():
            if count % 500 == 0:
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                robot.reset()

                robot.update(sim_dt)
                jacobian_b, ee_pose_b, root_pose_w, ee_pose_w, joint_pos = update_states(robot, ee_frame_idx, arm_joint_ids)

                active_goal_idx = current_goal_idx
                moving_to_waypoint = args_cli.enable_waypoint
                ee_target_pose_b[:] = ee_waypoint_pose_set_b[active_goal_idx] if moving_to_waypoint else ee_goal_pose_set_b[active_goal_idx]
                print("------------------- new eef goal --------------------")
                ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
                )
                ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

                diff_ik.reset()
                ik_commands[:] = ee_target_pose_b
                diff_ik.set_command(ik_commands)
                joint_pos_des = diff_ik.compute(ee_pose_b[:, 0:3], ee_pose_b[:, 3:7], jacobian_b, joint_pos)
                x_cmd_prev_b = ee_pose_b[:, 0:3].clone()
                current_goal_idx = (current_goal_idx + 1) % len(ee_goal_pose_set_b)

                contact_active[:] = False
                prev_contact_active[:] = False
                contact_confirm_counter.zero_()
                contact_latch_step.fill_(-1)
                force_filt_n.zero_()
                admittance_offset.zero_()
                admittance_velocity.zero_()
                admittance_acceleration.zero_()
                waypoint_hold_counter = 0
                steps_since_reset = 0
            else:
                jacobian_b, ee_pose_b, root_pose_w, ee_pose_w, joint_pos = update_states(robot, ee_frame_idx, arm_joint_ids)

                # Measured normal force (robot-on-wall, compression positive)
                if ft_body_idx is not None:
                    force_hand_b = robot.data.body_incoming_joint_wrench_b[:, ft_body_idx, 0:3]
                    hand_quat_w = robot.data.body_quat_w[:, ft_body_idx]
                    force_w = quat_apply(hand_quat_w, force_hand_b)
                    force_root_b = quat_apply_inverse(robot.data.root_quat_w, force_w)
                    f_ext_n_raw = torch.sum(force_root_b * contact_axis_b.unsqueeze(0), dim=-1)
                else:
                    f_ext_n_raw = torch.zeros(scene.num_envs, device=sim.device)

                # LPF: y_k = alpha*x_k + (1-alpha)*y_{k-1}
                force_filt_n = args_cli.force_filter_alpha * f_ext_n_raw + (1.0 - args_cli.force_filter_alpha) * force_filt_n

                # Contact detection only during final-goal approach and after delay.
                if (not moving_to_waypoint) and (steps_since_reset >= args_cli.contact_detection_delay_steps):
                    over_thresh = force_filt_n > args_cli.contact_force_threshold
                    contact_confirm_counter = torch.where(
                        over_thresh, contact_confirm_counter + 1, torch.zeros_like(contact_confirm_counter)
                    )
                    contact_active = torch.logical_or(contact_active, contact_confirm_counter >= args_cli.contact_confirm_steps)
                else:
                    contact_confirm_counter.zero_()

                # Waypoint -> final goal transition
                if moving_to_waypoint and (not torch.any(contact_active)):
                    waypoint_pos_err = torch.norm(ee_pose_b[:, 0:3] - ee_target_pose_b[:, 0:3], dim=-1)
                    if torch.all(waypoint_pos_err < args_cli.waypoint_switch_pos_thresh):
                        waypoint_hold_counter += 1
                    else:
                        waypoint_hold_counter = 0
                    if waypoint_hold_counter >= args_cli.waypoint_hold_steps:
                        moving_to_waypoint = False
                        ee_target_pose_b[:] = ee_goal_pose_set_b[active_goal_idx]
                        ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
                            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
                        )
                        ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

                # Desired force ramp: 0 before contact, smooth ramp after latch.
                newly_latched = torch.logical_and(contact_active, ~prev_contact_active)
                contact_latch_step = torch.where(newly_latched, torch.full_like(contact_latch_step, count), contact_latch_step)

                if args_cli.force_ramp_time > 0.0:
                    elapsed = torch.clamp((count - contact_latch_step + 1).to(torch.float32) * sim_dt, min=0.0)
                    ramp = torch.clamp(elapsed / args_cli.force_ramp_time, 0.0, 1.0)
                else:
                    ramp = torch.ones(scene.num_envs, device=sim.device)
                f_des_n = torch.where(contact_active, args_cli.desired_contact_force * ramp, torch.zeros_like(force_filt_n))

                f_err_n = f_des_n - force_filt_n

                # Admittance update (1D) with active clamps.
                admittance_acceleration = torch.where(
                    contact_active,
                    (f_err_n - args_cli.admittance_B * admittance_velocity - args_cli.admittance_K * admittance_offset)
                    / args_cli.admittance_M,
                    torch.zeros_like(admittance_acceleration),
                )
                admittance_velocity = torch.where(
                    contact_active,
                    admittance_velocity + admittance_acceleration * sim_dt,
                    torch.zeros_like(admittance_velocity),
                )
                admittance_velocity = torch.clamp(
                    admittance_velocity, -args_cli.max_admittance_velocity, args_cli.max_admittance_velocity
                )
                admittance_offset = torch.where(
                    contact_active,
                    admittance_offset + admittance_velocity * sim_dt,
                    torch.zeros_like(admittance_offset),
                )
                admittance_offset = torch.clamp(admittance_offset, -args_cli.max_admittance_offset, args_cli.max_admittance_offset)

                # Position command: nominal + normal-axis admittance offset.
                x_curr_b = ee_pose_b[:, 0:3]
                x_nominal_b = ee_target_pose_b[:, 0:3]
                x_delta_b = x_curr_b - x_nominal_b
                x_n = torch.sum(x_delta_b * contact_axis_b.unsqueeze(0), dim=-1)

                x_cmd_b = x_nominal_b + contact_axis_b.unsqueeze(0) * admittance_offset.unsqueeze(-1)

                # Hold non-contact components as much as possible.
                x_delta_contact_b = x_n.unsqueeze(-1) * contact_axis_b.unsqueeze(0)
                x_delta_non_contact_b = x_delta_b - x_delta_contact_b
                non_contact_correction_b = -args_cli.non_contact_hold_gain * x_delta_non_contact_b
                non_contact_correction_b = torch.clamp(
                    non_contact_correction_b,
                    -args_cli.max_non_contact_correction,
                    args_cli.max_non_contact_correction,
                )
                x_cmd_b = torch.where(contact_active.unsqueeze(-1), x_cmd_b + non_contact_correction_b, x_cmd_b)

                # Reduce final pre-contact approach speed for stable contact onset.
                if (not moving_to_waypoint) and (not torch.any(contact_active)):
                    cmd_delta = x_cmd_b - x_cmd_prev_b
                    cmd_delta_norm = torch.linalg.norm(cmd_delta, dim=-1, keepdim=True).clamp_min(1.0e-9)
                    max_step = args_cli.max_final_approach_speed * sim_dt
                    scale = torch.clamp(max_step / cmd_delta_norm, max=1.0)
                    x_cmd_b = x_cmd_prev_b + cmd_delta * scale

                # Fixed orientation from nominal target.
                ik_commands[:, 0:3] = x_cmd_b
                ik_commands[:, 3:7] = ee_target_pose_b[:, 3:7]
                diff_ik.set_command(ik_commands)
                joint_pos_des = diff_ik.compute(ee_pose_b[:, 0:3], ee_pose_b[:, 3:7], jacobian_b, joint_pos)
                x_cmd_prev_b = x_cmd_b.clone()

                if args_cli.debug_print_every > 0 and (count % args_cli.debug_print_every == 0):
                    phase = "WAYPOINT" if moving_to_waypoint else "FINAL"
                    print(
                        "[ADM] "
                        f"phase={phase} "
                        f"contact={bool(contact_active[0].item())} "
                        f"Fext={float(force_filt_n[0].item()):.3f}N "
                        f"Fdes={float(f_des_n[0].item()):.3f}N "
                        f"Ferr={float(f_err_n[0].item()):.3f}N "
                        f"x={float(admittance_offset[0].item()):.5f}m "
                        f"xdot={float(admittance_velocity[0].item()):.5f}m/s "
                        f"xddot={float(admittance_acceleration[0].item()):.5f}m/s2 "
                        f"xcmd=({float(x_cmd_b[0, 0].item()):.4f}, {float(x_cmd_b[0, 1].item()):.4f}, {float(x_cmd_b[0, 2].item()):.4f}) "
                        f"xcurr=({float(x_curr_b[0, 0].item()):.4f}, {float(x_curr_b[0, 1].item()):.4f}, {float(x_curr_b[0, 2].item()):.4f})"
                    )

                prev_contact_active = contact_active.clone()

            # Apply command
            robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()

            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

            sim.step(render=True)
            robot.update(sim_dt)
            scene.update(sim_dt)
            observer_camera.update(sim_dt)

            # CSV row
            ee_pos_b_env0 = ee_pose_b[0, 0:3].detach().cpu().tolist()
            ee_goal_pos_b_env0 = ee_target_pose_b[0, 0:3].detach().cpu().tolist()
            x_cmd_b_env0 = x_cmd_b[0].detach().cpu().tolist() if "x_cmd_b" in locals() else ee_goal_pos_b_env0

            if ft_writer is not None and ft_csv_file is not None:
                phase_val = "WAYPOINT" if moving_to_waypoint else "FINAL"
                ft_writer.writerow(
                    [
                        datetime.now().isoformat(),
                        count,
                        ee_pos_b_env0[0],
                        ee_pos_b_env0[1],
                        ee_pos_b_env0[2],
                        ee_goal_pos_b_env0[0],
                        ee_goal_pos_b_env0[1],
                        ee_goal_pos_b_env0[2],
                        x_cmd_b_env0[0],
                        x_cmd_b_env0[1],
                        x_cmd_b_env0[2],
                        float(x_n[0].item()) if "x_n" in locals() else 0.0,
                        float(force_filt_n[0].item()) if "force_filt_n" in locals() else 0.0,
                        float(f_ext_n_raw[0].item()) if "f_ext_n_raw" in locals() else 0.0,
                        float(force_filt_n[0].item()) if "force_filt_n" in locals() else 0.0,
                        float(f_des_n[0].item()) if "f_des_n" in locals() else 0.0,
                        float(f_err_n[0].item()) if "f_err_n" in locals() else 0.0,
                        float(admittance_offset[0].item()),
                        float(admittance_velocity[0].item()),
                        float(admittance_acceleration[0].item()),
                        int(contact_active[0].item()),
                        phase_val,
                    ]
                )
                ft_csv_file.flush()

            if video_writer is not None:
                rgb_frame = observer_camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                if rgb_frame.dtype != "uint8":
                    rgb_frame = (rgb_frame * 255.0).clip(0, 255).astype("uint8")
                video_writer.append_data(rgb_frame)
                if args_cli.record_length > 0 and (count + 1) >= args_cli.record_length:
                    break

            count += 1
            steps_since_reset += 1

    finally:
        if ft_csv_file is not None:
            ft_csv_file.close()
        if video_writer is not None:
            video_writer.close()
            print(f"[INFO] Saved video: {video_path}")
        if args_cli.log and ft_log_path is not None and ft_log_path.exists():
            plot_script = REPO_ROOT / "scripts" / "plot_data.py"
            if plot_script.exists():
                print(f"[INFO] Generating plot from log: {ft_log_path}")
                subprocess.run([sys.executable, str(plot_script), str(ft_log_path)], check=False)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    env0_origin = scene.env_origins[0]
    main_cam_eye = (env0_origin + torch.tensor([-0.25, 1.8, 1.55], device=sim.device)).tolist()
    main_cam_target = (env0_origin + torch.tensor([0.58, 0.0, 0.86], device=sim.device)).tolist()
    sim.set_camera_view(main_cam_eye, main_cam_target)

    sim.reset()
    print("[INFO] Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
