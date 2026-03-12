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
parser.add_argument(
    "--log_steps",
    type=int,
    default=1500,
    help="With --log, stop run after this many simulation steps. Set <=0 for full run.",
)
parser.add_argument(
    "--record",
    type=int,
    nargs="?",
    const=0,
    default=None,
    help="With --log, enable video recording. Optionally pass number of steps; omit value for full run.",
)
# Mode selection (replaces old --soft switch).
parser.add_argument(
    "--mode",
    type=str,
    choices=["rigid", "soft", "compliant"],
    default="rigid",
    help="Environment mode: rigid wall, deformable soft wall, or rigid wall with compliant contact.",
)
parser.add_argument("--soft", action="store_true", default=False, help=argparse.SUPPRESS)
parser.add_argument("--youngs_modulus", type=float, default=5e3, help="Young's modulus for deformable wall in --mode soft.")
parser.add_argument(
    "--compliant_contact_stiffness",
    type=float,
    default=1e6,
    help="Compliant contact stiffness for rigid wall in --mode compliant.",
)
parser.add_argument(
    "--compliant_contact_damping",
    type=float,
    default=1e3,
    help="Compliant contact damping for rigid wall in --mode compliant.",
)

# Required baseline knobs
parser.add_argument("--desired_contact_force", type=float, default=10.0, help="Desired normal force (N).")
parser.add_argument("--admittance_M", type=float, default=2.0, help="Admittance virtual mass.")
parser.add_argument("--admittance_B", type=float, default=130.0, help="Admittance virtual damping.")
parser.add_argument("--admittance_K", type=float, default=5.0, help="Admittance virtual stiffness.")
parser.add_argument("--contact_force_threshold", type=float, default=1.0, help="Contact threshold (N).")
parser.add_argument(
    "--soft_contact_pos_err_threshold",
    type=float,
    default=0.008,
    help="For --mode soft, latch contact when ||x_curr - x_goal_final|| is below this (m).",
)
parser.add_argument("--contact_confirm_steps", type=int, default=1, help="Consecutive threshold hits to latch contact.")
parser.add_argument(
    "--force_ramp_time",
    type=float,
    default=0.5,
    help="Ramp time (s) from 0 to desired force. Set 0 for a pure force step.",
)
parser.add_argument("--force_filter_alpha", type=float, default=0.075, help="LPF alpha for measured force [0,1].")
parser.add_argument("--max_admittance_offset", type=float, default=0.08, help="Clamp for admittance offset (m).")
parser.add_argument("--max_admittance_velocity", type=float, default=0.1, help="Clamp for admittance velocity (m/s).")
parser.add_argument("--debug_print_every", type=int, default=20, help="Print every N steps (0 disables).")
parser.add_argument(
    "--enable_tracking_anti_windup",
    action="store_true",
    default=False,
    help="Enable anti-windup: integrate admittance only when |tracking_error_n| < track_err_limit.",
)
parser.add_argument(
    "--track_err_limit",
    type=float,
    default=0.005,
    help="Normal-axis tracking error limit (m) used by anti-windup gate.",
)
parser.add_argument(
    "--disable_non_contact_correction",
    action="store_true",
    default=False,
    help="Disable non-contact correction term for A/B debugging.",
)
parser.add_argument(
    "--max_final_approach_speed",
    type=float,
    default=0.03,
    help="Max Cartesian speed (m/s) for final pre-contact approach command. Set <=0 to disable limiting.",
)

# Waypoint / robustness helpers
parser.add_argument("--contact_detection_delay_steps", type=int, default=20, help="Delay before contact detection after reset.")
parser.add_argument("--waypoint_offset", type=float, default=0.01, help="Waypoint retreat along -contact-axis (m).")
parser.add_argument("--wall_penetration_depth", type=float, default=0.0, help="Final goal penetration depth into wall along contact axis (m).")
parser.add_argument("--waypoint_switch_pos_thresh", type=float, default=0.008, help="Waypoint switch threshold (m).")
parser.add_argument("--waypoint_hold_steps", type=int, default=15, help="Consecutive waypoint-hold steps.")
parser.add_argument("--enable_waypoint", action="store_true", default=True, help="Enable pre-contact waypoint stage.")
parser.add_argument("--non_contact_hold_gain", type=float, default=4.0, help="Gain for non-contact-axis hold.")
parser.add_argument(
    "--max_non_contact_correction", type=float, default=0.03, help="Per-axis clamp for non-contact correction (m)."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Temporary backward compatibility for older scripts using --soft.
if args_cli.soft:
    print("[WARN] --soft is deprecated. Use --mode soft.")
    args_cli.mode = "soft"

# Baseline is intentionally single-env for clarity.
if args_cli.num_envs != 1:
    print(f"[WARN] Forcing num_envs=1 for baseline (requested {args_cli.num_envs}).")
    args_cli.num_envs = 1
if args_cli.record is not None and not args_cli.log:
    print("[WARN] --record is ignored unless --log is enabled.")

# Clamp alpha into [0, 1]
args_cli.force_filter_alpha = max(0.0, min(1.0, args_cli.force_filter_alpha))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Delayed imports after app launch
# pylint: disable=wrong-import-position
import carb  # noqa: E402
import csv  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg, DeformableObject, DeformableObjectCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat, quat_apply, quat_apply_inverse, subtract_frame_transforms

from source.franka import FRANKA_3_HIGH_PD_CFG  # noqa: E402


SOFT_WALL_INIT_POS = (0.50, -0.1, 0.03)
SOFT_WALL_INIT_ROT = (0.70710678, 0.0, -0.70710678, 0.0)
RIGID_WALL_INIT_POS = (0.50, -0.3, 0.85)
RIGID_WALL_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


def _enable_fractional_cutout_opacity():
    """Enable RTX fractional cutout opacity using type-safe settings write."""
    carb_settings = carb.settings.get_settings()
    fractional_cutout_path = "/rtx/raytracing/fractionalCutoutOpacity"
    current_val = carb_settings.get(fractional_cutout_path)

    if isinstance(current_val, bool):
        carb_settings.set_bool(fractional_cutout_path, True)
    elif isinstance(current_val, int):
        carb_settings.set_int(fractional_cutout_path, 1)
    elif isinstance(current_val, float):
        carb_settings.set_float(fractional_cutout_path, 1.0)
    else:
        carb_settings.set(fractional_cutout_path, 1)

    print(f"[INFO] Enabled fractional cutout opacity via setting: {fractional_cutout_path}")


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Wall-contact baseline scene (mode: rigid, soft, compliant)."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Mode-specific wall creation.
    if args_cli.mode == "soft":
        # Legacy deformable path retained for reference testing.
        soft_wall = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/SoftWall",
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.06, 0.8, 0.8),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.005),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0), opacity=0.5),
                physics_material=sim_utils.DeformableBodyMaterialCfg(
                    density=5000.0,
                    youngs_modulus=args_cli.youngs_modulus,
                    poissons_ratio=0.45,
                    dynamic_friction=0.7,
                ),
                physics_material_path="material",
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=SOFT_WALL_INIT_POS, rot=SOFT_WALL_INIT_ROT),
        )
    elif args_cli.mode == "compliant":
        # Rigid wall with compliant contact parameters for stiffness/damping sweeps.
        rigid_wall = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/RigidWall",
            spawn=sim_utils.CuboidCfg(
                size=(0.06, 0.8, 0.8),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0), opacity=1.0),
                # Compliant-contact parameter application for rigid wall.
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9,
                    dynamic_friction=0.7,
                    restitution=0.0,
                    compliant_contact_stiffness=args_cli.compliant_contact_stiffness,
                    compliant_contact_damping=args_cli.compliant_contact_damping,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=RIGID_WALL_INIT_POS, rot=RIGID_WALL_INIT_ROT),
        )
    else:
        # Default rigid wall mode.
        rigid_wall = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/RigidWall",
            spawn=sim_utils.CuboidCfg(
                size=(0.06, 0.8, 0.8),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0), opacity=1.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7, restitution=0.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=RIGID_WALL_INIT_POS, rot=RIGID_WALL_INIT_ROT),
        )

    robot = FRANKA_3_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["franka_shoulder"].stiffness = 1200.0
    robot.actuators["franka_shoulder"].damping = 80.0
    robot.actuators["franka_forearm"].stiffness = 1200.0
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


def _find_block_eight_corner_vertex_ids(
    nodal_pos_w: torch.Tensor,
    block_center_b: torch.Tensor,
    block_quat_b: torch.Tensor,
    block_size_xyz: tuple[float, float, float],
) -> torch.Tensor:
    """Find unique closest simulation vertices to all 8 cuboid corners."""
    block_rot_mat_b = matrix_from_quat(block_quat_b.unsqueeze(0)).squeeze(0)
    nodal_pos_local = torch.matmul((nodal_pos_w - block_center_b).to(block_rot_mat_b.dtype), block_rot_mat_b)

    hx = 0.5 * block_size_xyz[0]
    hy = 0.5 * block_size_xyz[1]
    hz = 0.5 * block_size_xyz[2]
    corner_targets_local = torch.tensor(
        [
            [hx, hy, hz],
            [hx, hy, -hz],
            [hx, -hy, hz],
            [hx, -hy, -hz],
            [-hx, hy, hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [-hx, -hy, -hz],
        ],
        device=nodal_pos_w.device,
        dtype=nodal_pos_w.dtype,
    )
    distances = torch.sum((nodal_pos_local.unsqueeze(2) - corner_targets_local.unsqueeze(0).unsqueeze(0)) ** 2, dim=-1)
    num_envs, num_vertices, _ = distances.shape
    corner_vertex_ids = torch.empty((num_envs, 8), dtype=torch.long, device=nodal_pos_w.device)
    for env_id in range(num_envs):
        d = distances[env_id].clone()
        assigned_vertices: set[int] = set()
        assigned_corners: set[int] = set()
        for _ in range(8):
            best_val = None
            best_v = -1
            best_c = -1
            for c in range(8):
                if c in assigned_corners:
                    continue
                for v in range(num_vertices):
                    if v in assigned_vertices:
                        continue
                    val = d[v, c].item()
                    if best_val is None or val < best_val:
                        best_val = val
                        best_v = v
                        best_c = c
            assigned_vertices.add(best_v)
            assigned_corners.add(best_c)
            corner_vertex_ids[env_id, best_c] = best_v
    return corner_vertex_ids


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
    is_soft_mode = args_cli.mode == "soft"
    is_compliant_mode = args_cli.mode == "compliant"
    soft_wall: DeformableObject | None = scene["soft_wall"] if is_soft_mode else None

    ee_frame_name = "fr3_leftfinger"
    arm_joint_names = ["fr3_joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]
    env_origins = scene.env_origins

    # Observer camera + main camera matching viewpoint.
    camera_positions = env_origins + torch.tensor([0.7, 0.8, 0.7], device=sim.device, dtype=env_origins.dtype)
    camera_targets = env_origins + torch.tensor([0.5, -0.1, 0.0], device=sim.device, dtype=env_origins.dtype)
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
        logs_root = REPO_ROOT / "logs" / f"admittance_baseline_floor_{args_cli.mode}"
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
                "mode",
                "youngs_modulus_pa",
                "wall_compliant_contact_stiffness",
                "wall_compliant_contact_damping",
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
                "f_world_z_raw",
                "f_contact_axis_raw",
                "f_contact_axis_filt",
                "f_compression_pos_raw",
                "f_compression_pos_filt",
                "f_des_n",
                "f_err_n",
                "admittance_offset",
                "admittance_velocity",
                "admittance_acceleration",
                "contact_active",
                "phase",
                "x_cmd_n",
                "x_curr_n",
                "tracking_error_n",
                "tracking_error_n_prev",
                "admittance_integrate_enabled",
                "non_contact_correction_mag",
                "ik_target_delta_norm",
                "ik_target_out_of_limits",
                "x_cmd_step_clipped",
                "ik_cmd_b_x",
                "ik_cmd_b_y",
                "ik_cmd_b_z",
                "joint_pos_des_0",
                "joint_pos_des_1",
                "joint_pos_des_2",
                "joint_pos_des_3",
                "joint_pos_des_4",
                "joint_pos_des_5",
                "joint_pos_des_6",
            ]
        )
        print(f"[INFO] FT log file: {ft_log_path}")
        print(f"[INFO] Mode: {args_cli.mode}")
        if is_soft_mode:
            print(f"[INFO] Young's modulus: {args_cli.youngs_modulus:.6g}")
        if is_compliant_mode:
            print(f"[INFO] Compliant contact stiffness: {args_cli.compliant_contact_stiffness:.6g}")
            print(f"[INFO] Compliant contact damping: {args_cli.compliant_contact_damping:.6g}")

    video_writer = None
    video_path = None
    if args_cli.log and run_dir is not None and args_cli.record is not None:
        video_path = run_dir / "admittance_baseline_floor_env0.mp4"
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

    # Wall geometry in root/body frame (matches SceneCfg)
    wall_init_pos = SOFT_WALL_INIT_POS if is_soft_mode else RIGID_WALL_INIT_POS
    wall_center_b = torch.tensor(wall_init_pos, device=sim.device)
    wall_init_rot = SOFT_WALL_INIT_ROT if is_soft_mode else RIGID_WALL_INIT_ROT
    wall_quat_b = torch.tensor(wall_init_rot, device=sim.device)
    wall_thickness = 0.06
    wall_rot_mat_b = matrix_from_quat(wall_quat_b.unsqueeze(0)).squeeze(0)
    wall_x_axis_b = wall_rot_mat_b[:, 0]
    wall_y_axis_b = wall_rot_mat_b[:, 1]
    wall_z_axis_b = wall_rot_mat_b[:, 2]

    # Contact axis convention by mode:
    # - rigid/compliant: near-side normal is -X, compression axis is +X
    # - soft: force contact axis to world -Z so positive admittance offset presses downward
    world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=sim.device)
    if is_soft_mode:
        wall_normal_out_b = world_z_axis
        contact_axis_b = -world_z_axis
    else:
        wall_normal_out_b = -wall_x_axis_b
        contact_axis_b = -wall_normal_out_b
    wall_surface_center_b = wall_center_b + 0.5 * wall_thickness * wall_normal_out_b

    # Soft mode uses floor support, so keep all deformable nodes free (no nodal constraints).
    wall_nodal_kinematic_target = None

    # Goal set:
    # - soft mode: center point plus two points in +Y
    # - rigid/compliant: three lateral points
    if is_soft_mode:
        wall_plane_offsets_yz = torch.tensor([[0.0, 0.0], [0.10, 0.0], [0.20, 0.0]], device=sim.device)
    else:
        wall_plane_offsets_yz = torch.tensor([[0.10, 0.0], [0.0, 0.0], [-0.10, 0.0]], device=sim.device)
    effective_wall_penetration_depth = args_cli.wall_penetration_depth
    wall_contact_center_b = wall_surface_center_b + effective_wall_penetration_depth * contact_axis_b
    ee_goal_pos_set_b = (
        wall_contact_center_b.unsqueeze(0)
        + wall_plane_offsets_yz[:, 0:1] * wall_y_axis_b.unsqueeze(0)
        + wall_plane_offsets_yz[:, 1:2] * wall_z_axis_b.unsqueeze(0)
    )
    if is_soft_mode:
        # Floor-contact soft mode: keep nominal goal slightly above the top surface to account for finger-frame offset.
        ee_goal_pos_set_b[:, 2] = 0.116
    # Keep fixed X target only for vertical-wall modes.
    if not is_soft_mode:
        ee_goal_pos_set_b[:, 0] = 0.5138
    num_goal_points = ee_goal_pos_set_b.shape[0]
    if is_soft_mode:
        # Make EE local +Z (blue axis) point to world -Z.
        ee_goal_quat_set_b = torch.tensor([0.0, 1.0, 0.0, 0.0], device=sim.device).repeat(num_goal_points, 1)
    else:
        ee_goal_quat_set_b = torch.tensor([0.0, 0.70710678, 0.0, 0.70710678], device=sim.device).repeat(
            num_goal_points, 1
        )
    ee_goal_pose_set_b = torch.cat([ee_goal_pos_set_b, ee_goal_quat_set_b], dim=-1)

    # Waypoint retreat is controlled directly by --waypoint_offset.
    effective_waypoint_offset = args_cli.waypoint_offset
    ee_waypoint_pos_set_b = ee_goal_pos_set_b - effective_waypoint_offset * contact_axis_b.unsqueeze(0)
    ee_waypoint_pose_set_b = torch.cat([ee_waypoint_pos_set_b, ee_goal_quat_set_b], dim=-1)

    sim_dt = sim.get_physics_dt()
    robot.update(sim_dt)

    jacobian_b, ee_pose_b, root_pose_w, ee_pose_w, joint_pos = update_states(robot, ee_frame_idx, arm_joint_ids)
    arm_joint_limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids, :]

    # Controller state
    current_goal_idx = 0
    active_goal_idx = 0
    moving_to_waypoint = True
    waypoint_hold_counter = 0

    contact_active = torch.zeros(scene.num_envs, dtype=torch.bool, device=sim.device)
    prev_contact_active = torch.zeros_like(contact_active)
    contact_confirm_counter = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    contact_latch_step = torch.full((scene.num_envs,), -1, dtype=torch.long, device=sim.device)

    f_contact_axis_filt = torch.zeros(scene.num_envs, device=sim.device)
    f_compression_pos_filt = torch.zeros(scene.num_envs, device=sim.device)
    admittance_offset = torch.zeros(scene.num_envs, device=sim.device)
    admittance_velocity = torch.zeros(scene.num_envs, device=sim.device)
    admittance_acceleration = torch.zeros(scene.num_envs, device=sim.device)

    ik_commands = torch.zeros(scene.num_envs, diff_ik.action_dim, device=sim.device)
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)
    joint_pos_des = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)
    x_cmd_prev_b = ee_pose_b[:, 0:3].clone()
    # Debug instrumentation buffers for command-path and tracking diagnostics.
    x_cmd_b = ee_pose_b[:, 0:3].clone()
    x_cmd_n = torch.zeros(scene.num_envs, device=sim.device)
    x_curr_n = torch.zeros(scene.num_envs, device=sim.device)
    tracking_error_n = torch.zeros(scene.num_envs, device=sim.device)
    tracking_error_n_prev = torch.zeros(scene.num_envs, device=sim.device)
    non_contact_correction_mag = torch.zeros(scene.num_envs, device=sim.device)
    admittance_integrate_enabled = torch.zeros(scene.num_envs, dtype=torch.bool, device=sim.device)
    ik_target_delta_norm = torch.zeros(scene.num_envs, device=sim.device)
    ik_target_out_of_limits = torch.zeros(scene.num_envs, dtype=torch.bool, device=sim.device)
    x_cmd_step_clipped = torch.zeros(scene.num_envs, dtype=torch.bool, device=sim.device)

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
                x_cmd_b = ee_pose_b[:, 0:3].clone()
                current_goal_idx = (current_goal_idx + 1) % len(ee_goal_pose_set_b)

                contact_active[:] = False
                prev_contact_active[:] = False
                contact_confirm_counter.zero_()
                contact_latch_step.fill_(-1)
                f_contact_axis_filt.zero_()
                f_compression_pos_filt.zero_()
                admittance_offset.zero_()
                admittance_velocity.zero_()
                admittance_acceleration.zero_()
                waypoint_hold_counter = 0
                steps_since_reset = 0
            else:
                jacobian_b, ee_pose_b, root_pose_w, ee_pose_w, joint_pos = update_states(robot, ee_frame_idx, arm_joint_ids)

                # body_incoming_joint_wrench_b is a joint reaction wrench in a body-related frame.
                # We explicitly derive world-Z, contact-axis projection, and compression-positive force.
                if ft_body_idx is not None:
                    force_hand_b = robot.data.body_incoming_joint_wrench_b[:, ft_body_idx, 0:3]
                    hand_quat_w = robot.data.body_quat_w[:, ft_body_idx]
                    force_w = quat_apply(hand_quat_w, force_hand_b)
                    force_root_b = quat_apply_inverse(robot.data.root_quat_w, force_w)
                    f_world_z_raw = force_w[:, 2]
                    f_contact_axis_raw = torch.sum(force_root_b * contact_axis_b.unsqueeze(0), dim=-1)
                else:
                    f_world_z_raw = torch.zeros(scene.num_envs, device=sim.device)
                    f_contact_axis_raw = torch.zeros(scene.num_envs, device=sim.device)

                # LPF: y_k = alpha*x_k + (1-alpha)*y_{k-1}
                f_contact_axis_filt = (
                    args_cli.force_filter_alpha * f_contact_axis_raw + (1.0 - args_cli.force_filter_alpha) * f_contact_axis_filt
                )

                # Compression-positive force used by contact logic and admittance force error.
                f_compression_pos_raw = -f_contact_axis_raw
                f_compression_pos_filt = (
                    args_cli.force_filter_alpha * f_compression_pos_raw
                    + (1.0 - args_cli.force_filter_alpha) * f_compression_pos_filt
                )

                # Contact detection only during final-goal approach and after delay.
                if (not moving_to_waypoint) and (steps_since_reset >= args_cli.contact_detection_delay_steps):
                    if is_soft_mode:
                        final_goal_pos_err = torch.norm(
                            ee_pose_b[:, 0:3] - ee_goal_pose_set_b[active_goal_idx, 0:3].unsqueeze(0), dim=-1
                        )
                        over_thresh_force = f_compression_pos_filt > args_cli.contact_force_threshold
                        over_thresh_pos = final_goal_pos_err < args_cli.soft_contact_pos_err_threshold
                        over_thresh = torch.logical_or(over_thresh_force, over_thresh_pos)
                    else:
                        over_thresh = f_compression_pos_filt > args_cli.contact_force_threshold
                    contact_confirm_counter = torch.where(
                        over_thresh, contact_confirm_counter + 1, torch.zeros_like(contact_confirm_counter)
                    )
                    contact_active = torch.logical_or(contact_active, contact_confirm_counter >= args_cli.contact_confirm_steps)
                else:
                    contact_confirm_counter.zero_()

                # Waypoint -> final goal transition
                if moving_to_waypoint and (not torch.any(contact_active)):
                    if is_soft_mode:
                        # In soft mode, use normal-axis waypoint error to avoid Y/Z contact-induced bias.
                        waypoint_delta_b = ee_pose_b[:, 0:3] - ee_target_pose_b[:, 0:3]
                        waypoint_pos_err = torch.abs(torch.sum(waypoint_delta_b * contact_axis_b.unsqueeze(0), dim=-1))
                    else:
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
                f_des_n = torch.where(
                    contact_active, args_cli.desired_contact_force * ramp, torch.zeros_like(f_compression_pos_filt)
                )

                # Desired-vs-measured force error in compression-positive convention.
                f_err_n = f_des_n - f_compression_pos_filt

                x_curr_b = ee_pose_b[:, 0:3]
                x_nominal_b = ee_target_pose_b[:, 0:3]
                x_delta_b = x_curr_b - x_nominal_b
                x_curr_n = torch.sum(x_delta_b * contact_axis_b.unsqueeze(0), dim=-1)
                x_cmd_n_prev = torch.sum((x_cmd_prev_b - x_nominal_b) * contact_axis_b.unsqueeze(0), dim=-1)
                tracking_error_n_prev = x_cmd_n_prev - x_curr_n

                # Optional anti-windup: integrate admittance only while Cartesian normal tracking is good.
                admittance_integrate_enabled = contact_active.clone()
                if args_cli.enable_tracking_anti_windup:
                    admittance_integrate_enabled = torch.logical_and(
                        contact_active, torch.abs(tracking_error_n_prev) < args_cli.track_err_limit
                    )

                # Admittance update (1D) with active clamps.
                admittance_acceleration = torch.where(
                    admittance_integrate_enabled,
                    (f_err_n - args_cli.admittance_B * admittance_velocity - args_cli.admittance_K * admittance_offset)
                    / args_cli.admittance_M,
                    torch.zeros_like(admittance_acceleration),
                )
                vel_candidate = admittance_velocity + admittance_acceleration * sim_dt
                vel_candidate = torch.clamp(vel_candidate, -args_cli.max_admittance_velocity, args_cli.max_admittance_velocity)
                admittance_velocity = torch.where(
                    contact_active,
                    torch.where(admittance_integrate_enabled, vel_candidate, torch.zeros_like(admittance_velocity)),
                    torch.zeros_like(admittance_velocity),
                )
                offset_candidate = admittance_offset + admittance_velocity * sim_dt
                offset_candidate = torch.clamp(offset_candidate, -args_cli.max_admittance_offset, args_cli.max_admittance_offset)
                admittance_offset = torch.where(
                    contact_active,
                    torch.where(admittance_integrate_enabled, offset_candidate, admittance_offset),
                    torch.zeros_like(admittance_offset),
                )

                # Position command: nominal + normal-axis admittance offset.

                x_cmd_b = x_nominal_b + contact_axis_b.unsqueeze(0) * admittance_offset.unsqueeze(-1)

                # Hold non-contact components as much as possible.
                x_delta_contact_b = x_curr_n.unsqueeze(-1) * contact_axis_b.unsqueeze(0)
                x_delta_non_contact_b = x_delta_b - x_delta_contact_b
                non_contact_correction_b = -args_cli.non_contact_hold_gain * x_delta_non_contact_b
                non_contact_correction_b = torch.clamp(
                    non_contact_correction_b,
                    -args_cli.max_non_contact_correction,
                    args_cli.max_non_contact_correction,
                )
                if is_soft_mode:
                    # Floor-contact mode: only allow corrective motion in XY.
                    non_contact_correction_b = torch.cat(
                        [non_contact_correction_b[:, 0:2], torch.zeros_like(non_contact_correction_b[:, 2:3])], dim=-1
                    )
                non_contact_correction_mag = torch.linalg.norm(non_contact_correction_b, dim=-1)
                if not args_cli.disable_non_contact_correction:
                    x_cmd_b = torch.where(contact_active.unsqueeze(-1), x_cmd_b + non_contact_correction_b, x_cmd_b)
                else:
                    non_contact_correction_mag = torch.zeros_like(non_contact_correction_mag)

                if is_soft_mode:
                    # Keep normal pressing strictly on world-Z in soft mode.
                    x_cmd_b[:, 2] = x_nominal_b[:, 2] + contact_axis_b[2] * admittance_offset

                # Reduce final pre-contact approach speed for stable contact onset.
                x_cmd_step_clipped = torch.zeros_like(x_cmd_step_clipped)
                if (not moving_to_waypoint) and (not torch.any(contact_active)):
                    if args_cli.max_final_approach_speed > 0.0:
                        cmd_delta = x_cmd_b - x_cmd_prev_b
                        cmd_delta_norm = torch.linalg.norm(cmd_delta, dim=-1, keepdim=True).clamp_min(1.0e-9)
                        max_step = args_cli.max_final_approach_speed * sim_dt
                        scale = torch.clamp(max_step / cmd_delta_norm, max=1.0)
                        x_cmd_step_clipped = torch.squeeze(scale < 1.0, dim=-1)
                        x_cmd_b = x_cmd_prev_b + cmd_delta * scale

                # Fixed orientation from nominal target.
                ik_commands[:, 0:3] = x_cmd_b
                ik_commands[:, 3:7] = ee_target_pose_b[:, 3:7]
                diff_ik.set_command(ik_commands)
                joint_pos_des = diff_ik.compute(ee_pose_b[:, 0:3], ee_pose_b[:, 3:7], jacobian_b, joint_pos)
                ik_target_delta_norm = torch.linalg.norm(joint_pos_des - joint_pos, dim=-1)
                lower = arm_joint_limits[:, :, 0]
                upper = arm_joint_limits[:, :, 1]
                ik_target_out_of_limits = torch.any(torch.logical_or(joint_pos_des < lower, joint_pos_des > upper), dim=-1)
                x_cmd_prev_b = x_cmd_b.clone()
                x_cmd_n = torch.sum((x_cmd_b - x_nominal_b) * contact_axis_b.unsqueeze(0), dim=-1)
                tracking_error_n = x_cmd_n - x_curr_n

                if args_cli.debug_print_every > 0 and (count % args_cli.debug_print_every == 0):
                    phase = "WAYPOINT" if moving_to_waypoint else "FINAL"
                    print(
                        "[ADM] "
                        f"phase={phase} "
                        f"contact={bool(contact_active[0].item())} "
                        f"FworldZ={float(f_world_z_raw[0].item()):.3f}N "
                        f"Faxis={float(f_contact_axis_raw[0].item()):.3f}N "
                        f"Fcomp={float(f_compression_pos_filt[0].item()):.3f}N "
                        f"Fdes={float(f_des_n[0].item()):.3f}N "
                        f"Ferr={float(f_err_n[0].item()):.3f}N "
                        f"x={float(admittance_offset[0].item()):.5f}m "
                        f"xcmd_n={float(x_cmd_n[0].item()):.5f}m "
                        f"xcurr_n={float(x_curr_n[0].item()):.5f}m "
                        f"track_n={float(tracking_error_n[0].item()):.5f}m "
                        f"xdot={float(admittance_velocity[0].item()):.5f}m/s "
                        f"xddot={float(admittance_acceleration[0].item()):.5f}m/s2 "
                        f"aw={bool(admittance_integrate_enabled[0].item())} "
                        f"nc_mag={float(non_contact_correction_mag[0].item()):.5f} "
                        f"ik_lim={bool(ik_target_out_of_limits[0].item())} "
                        f"xcmd=({float(x_cmd_b[0, 0].item()):.4f}, {float(x_cmd_b[0, 1].item()):.4f}, {float(x_cmd_b[0, 2].item()):.4f}) "
                        f"xcurr=({float(x_curr_b[0, 0].item()):.4f}, {float(x_curr_b[0, 1].item()):.4f}, {float(x_curr_b[0, 2].item()):.4f})"
                    )

                prev_contact_active = contact_active.clone()

            # Apply command
            robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()
            if is_soft_mode and soft_wall is not None and wall_nodal_kinematic_target is not None:
                soft_wall.write_nodal_kinematic_target_to_sim(wall_nodal_kinematic_target)
                soft_wall.write_data_to_sim()

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
                        args_cli.mode,
                        float(args_cli.youngs_modulus) if is_soft_mode else float("nan"),
                        float(args_cli.compliant_contact_stiffness) if is_compliant_mode else float("nan"),
                        float(args_cli.compliant_contact_damping) if is_compliant_mode else float("nan"),
                        ee_pos_b_env0[0],
                        ee_pos_b_env0[1],
                        ee_pos_b_env0[2],
                        ee_goal_pos_b_env0[0],
                        ee_goal_pos_b_env0[1],
                        ee_goal_pos_b_env0[2],
                        x_cmd_b_env0[0],
                        x_cmd_b_env0[1],
                        x_cmd_b_env0[2],
                        float(x_curr_n[0].item()) if "x_curr_n" in locals() else 0.0,
                        float(f_world_z_raw[0].item()) if "f_world_z_raw" in locals() else 0.0,
                        float(f_contact_axis_raw[0].item()) if "f_contact_axis_raw" in locals() else 0.0,
                        float(f_contact_axis_filt[0].item()) if "f_contact_axis_filt" in locals() else 0.0,
                        float(f_compression_pos_raw[0].item()) if "f_compression_pos_raw" in locals() else 0.0,
                        float(f_compression_pos_filt[0].item()) if "f_compression_pos_filt" in locals() else 0.0,
                        float(f_des_n[0].item()) if "f_des_n" in locals() else 0.0,
                        float(f_err_n[0].item()) if "f_err_n" in locals() else 0.0,
                        float(admittance_offset[0].item()),
                        float(admittance_velocity[0].item()),
                        float(admittance_acceleration[0].item()),
                        int(contact_active[0].item()),
                        phase_val,
                        float(x_cmd_n[0].item()) if "x_cmd_n" in locals() else 0.0,
                        float(x_curr_n[0].item()) if "x_curr_n" in locals() else 0.0,
                        float(tracking_error_n[0].item()) if "tracking_error_n" in locals() else 0.0,
                        float(tracking_error_n_prev[0].item()) if "tracking_error_n_prev" in locals() else 0.0,
                        int(admittance_integrate_enabled[0].item()) if "admittance_integrate_enabled" in locals() else 0,
                        float(non_contact_correction_mag[0].item()) if "non_contact_correction_mag" in locals() else 0.0,
                        float(ik_target_delta_norm[0].item()) if "ik_target_delta_norm" in locals() else 0.0,
                        int(ik_target_out_of_limits[0].item()) if "ik_target_out_of_limits" in locals() else 0,
                        int(x_cmd_step_clipped[0].item()) if "x_cmd_step_clipped" in locals() else 0,
                        x_cmd_b_env0[0],
                        x_cmd_b_env0[1],
                        x_cmd_b_env0[2],
                        float(joint_pos_des[0, 0].item()) if joint_pos_des.shape[1] > 0 else 0.0,
                        float(joint_pos_des[0, 1].item()) if joint_pos_des.shape[1] > 1 else 0.0,
                        float(joint_pos_des[0, 2].item()) if joint_pos_des.shape[1] > 2 else 0.0,
                        float(joint_pos_des[0, 3].item()) if joint_pos_des.shape[1] > 3 else 0.0,
                        float(joint_pos_des[0, 4].item()) if joint_pos_des.shape[1] > 4 else 0.0,
                        float(joint_pos_des[0, 5].item()) if joint_pos_des.shape[1] > 5 else 0.0,
                        float(joint_pos_des[0, 6].item()) if joint_pos_des.shape[1] > 6 else 0.0,
                    ]
                )
                ft_csv_file.flush()

            if video_writer is not None:
                rgb_frame = observer_camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                if rgb_frame.dtype != "uint8":
                    rgb_frame = (rgb_frame * 255.0).clip(0, 255).astype("uint8")
                video_writer.append_data(rgb_frame)
                if args_cli.record is not None and args_cli.record > 0 and (count + 1) >= args_cli.record:
                    break

            if args_cli.log and args_cli.log_steps > 0 and (count + 1) >= args_cli.log_steps:
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
                plot_title = f"Admittance Baseline Floor ({args_cli.mode})"
                subprocess.run([sys.executable, str(plot_script), str(ft_log_path), "--title", plot_title], check=False)


def main():
    # Enable translucency through RenderCfg.
    render_cfg = sim_utils.RenderCfg(enable_translucency=True)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, render=render_cfg)
    sim = sim_utils.SimulationContext(sim_cfg)
    _enable_fractional_cutout_opacity()

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
