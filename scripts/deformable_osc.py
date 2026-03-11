# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the operational space controller (OSC) with the simulator.

The OSC controller can be configured in different modes. It uses the dynamical quantities such as Jacobians and
mass matricescomputed by PhysX.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_osc.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the operational space controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--log", action="store_true", default=False, help="Enable FT CSV logging and video recording.")
parser.add_argument("--record_length", type=int, default=0, help="With --log, number of simulation steps to record. 0 means full run.")
parser.add_argument(
    "--youngs_modulus",
    type=float,
    default=2e5,
    help="Deformable block Young's modulus (Pa).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments 
args_cli = parser.parse_args()
# Always enable cameras for observer recording/monitoring.
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Delayed imports are intentional: Isaac Sim must be launched before loading these modules.
# pylint: disable=wrong-import-position
import csv  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg, DeformableObject  # noqa: E402
from isaaclab.assets import DeformableObjectCfg  # noqa: E402
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)  # noqa: E402

##
# Pre-defined configs
##
from source.franka import FRANKA_3_HIGH_PD_CFG  # isort:skip  # noqa: E402
# pylint: enable=wrong-import-position


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with a soft block."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Large deformable block.
    soft_block = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoftBlock",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.8, 1.5, 1.0),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.005,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=1.0),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                density=5000.0,
                youngs_modulus=args_cli.youngs_modulus,
                poissons_ratio=0.45,
                dynamic_friction=0.6,
            ),
            physics_material_path="material",
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.7), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    robot = FRANKA_3_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["franka_shoulder"].stiffness = 0.0
    robot.actuators["franka_shoulder"].damping = 0.0
    robot.actuators["franka_forearm"].stiffness = 0.0
    robot.actuators["franka_forearm"].damping = 0.0
    robot.spawn.rigid_props.disable_gravity = True

    # Observer camera for logging/recording.
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
    """Find unique closest simulation vertices to all 8 block corners for each environment."""
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
    distances = torch.sum(
        (nodal_pos_local.unsqueeze(2) - corner_targets_local.unsqueeze(0).unsqueeze(0)) ** 2,
        dim=-1,
    )
    # Enforce one-to-one assignment (8 unique vertices) per environment.
    num_envs, num_vertices, _ = distances.shape
    corner_vertex_ids = torch.empty((num_envs, 8), dtype=torch.long, device=nodal_pos_w.device)
    for env_id in range(num_envs):
        d = distances[env_id].clone()  # (num_vertices, 8)
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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
    """

    # Extract scene entities for readability.
    robot = scene["robot"]
    soft_block: DeformableObject = scene["soft_block"]
    observer_camera = scene["observer_camera"]

    # Obtain indices for the end-effector and arm joints
    ee_frame_name = "fr3_leftfinger"
    arm_joint_names = ["fr3_joint.*"]
    finger_joint_names = ["fr3_finger_joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]
    finger_joint_ids = robot.find_joints(finger_joint_names)[0]
    env_origins = scene.env_origins

    # Side view camera: move further right and yaw left to keep wall/object centered.
    camera_positions = env_origins + torch.tensor([-0.35, 3.0, 1.35], device=sim.device, dtype=env_origins.dtype)
    camera_targets = env_origins + torch.tensor([0.62, 0.0, 0.74], device=sim.device, dtype=env_origins.dtype)
    observer_camera.set_world_poses_from_view(camera_positions, camera_targets)

    # Resolve FT body index for the joint /World/envs/env_0/Robot/fr3/fr3_link8/fr3_hand_joint
    # by reading the incoming joint wrench on child body fr3_hand.
    ft_body_name = "fr3_hand"
    ft_body_idx = None
    try:
        ft_body_ids, _ = robot.find_bodies(ft_body_name)
        if len(ft_body_ids) > 0:
            ft_body_idx = int(ft_body_ids[0])
        else:
            print(f"[WARN] Body '{ft_body_name}' not found. FT logging will write empty values.")
    except Exception as exc:
        print(f"[WARN] Failed to resolve FT body index for '{ft_body_name}': {exc}.")

    # Setup optional logging (FT CSV + video).
    ft_csv_file = None
    ft_writer = None
    run_dir = None
    if args_cli.log:
        logs_root = REPO_ROOT / "logs" / "deformable_osc"
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
                "youngs_modulus_pa",
                "ee_pos_b_x",
                "ee_pos_b_y",
                "ee_pos_b_z",
                "ee_goal_pos_b_x",
                "ee_goal_pos_b_y",
                "ee_goal_pos_b_z",
                "ee_pos_err_b_x",
                "ee_pos_err_b_y",
                "ee_pos_err_b_z",
                "ee_pos_err_b_norm",
                "joint_path",
                "fx",
                "fy",
                "fz",
                "tx",
                "ty",
                "tz",
            ]
        )
        print(f"[INFO] FT log file: {ft_log_path}")
    print(f"[INFO] Young's modulus: {args_cli.youngs_modulus:.6g}")

    # Setup optional video writer.
    video_writer = None
    video_path = None
    if args_cli.log and run_dir is not None:
        video_path = run_dir / "deformable_osc_env0.mp4"
        fps = max(1, int(round(1.0 / sim.get_physics_dt())))
        video_writer = imageio.get_writer(video_path, fps=fps)
        print(f"[INFO] Recording video to: {video_path}")
        print(f"[INFO] Video FPS: {fps}")

    # Create the OSC
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_damping_ratio_task=1.0,
        contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        motion_control_axes_task=[1, 1, 0, 1, 1, 1],
        contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
        nullspace_control="position",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define targets for the arm
    # Build EE goals on one side face of a large deformable block (env/base frame).
    wall_center_b = torch.tensor([0.75, 0.0, 0.5], device=sim.device)
    wall_quat_b = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
    block_size_xyz = (0.8, 1.5, 1.0)
    wall_rot_mat_b = matrix_from_quat(wall_quat_b.unsqueeze(0)).squeeze(0)
    wall_x_axis_b = wall_rot_mat_b[:, 0]
    wall_y_axis_b = wall_rot_mat_b[:, 1]
    wall_z_axis_b = wall_rot_mat_b[:, 2]
    # Use the -X face (closer to robot) as contact face.
    wall_normal_b = -wall_x_axis_b
    wall_surface_center_b = wall_center_b - 0.5 * block_size_xyz[0] * wall_x_axis_b

    # Pin 8 corners of the block to reduce wobble.
    wall_nodal_kinematic_target = soft_block.data.nodal_kinematic_target.clone()
    block_corner_vertex_ids = _find_block_eight_corner_vertex_ids(
        nodal_pos_w=soft_block.data.default_nodal_state_w[..., :3],
        block_center_b=wall_center_b,
        block_quat_b=wall_quat_b,
        block_size_xyz=block_size_xyz,
    )

    # fr3_leftfinger is used as EE frame; target exactly on block surface along surface normal.
    wall_penetration_depth = 0.0
    wall_contact_center_b = wall_surface_center_b - wall_penetration_depth * wall_normal_b
    wall_tangent_1_b = wall_y_axis_b
    wall_tangent_2_b = wall_z_axis_b
    # Three distinct points on the side face plane (in meters, along face tangential axes).
    wall_plane_offsets_t = torch.tensor(
        [
            [0.00, 0.00],
            [0.00, 0.20],
            [-0.15, -0.20],
        ],
        device=sim.device,
    )
    ee_goal_pos_set_tilted_b = (
        wall_contact_center_b.unsqueeze(0)
        + wall_plane_offsets_t[:, 0:1] * wall_tangent_1_b.unsqueeze(0)
        + wall_plane_offsets_t[:, 1:2] * wall_tangent_2_b.unsqueeze(0)
    )
    # +Z (blue) points into block (+X), +X (red) points up (+Z).
    ee_goal_quat_set_tilted_b = torch.tensor([0.0, 0.70710678, 0.0, 0.70710678], device=sim.device).repeat(3, 1)
    ee_goal_pose_set_tilted_b = torch.cat([ee_goal_pos_set_tilted_b, ee_goal_quat_set_tilted_b], dim=-1)
    ee_goal_wrench_set_tilted_task = torch.tensor(
        [
            [0.0, 0.0, 20.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 20.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 20.0, 0.0, 0.0, 0.0],
        ],
        device=sim.device,
    )
    kp_set_task = torch.tensor(
        [
            [360.0, 360.0, 360.0, 360.0, 360.0, 360.0],
            [420.0, 420.0, 420.0, 420.0, 420.0, 420.0],
            [320.0, 320.0, 320.0, 320.0, 320.0, 320.0],
        ],
        device=sim.device,
    )
    ee_target_set = torch.cat([ee_goal_pose_set_tilted_b, ee_goal_wrench_set_tilted_task, kp_set_task], dim=-1)

    # Initialize block corner constraints once; do not reset the block during arm resets.
    wall_nodal_state = soft_block.data.default_nodal_state_w.clone()
    wall_nodal_kinematic_target[..., :3] = wall_nodal_state[..., :3]
    wall_nodal_kinematic_target[..., 3] = 1.0  # free all nodes except pinned corners
    env_ids = torch.arange(scene.num_envs, device=sim.device)
    for i in range(8):
        vertex_ids = block_corner_vertex_ids[:, i]
        wall_nodal_kinematic_target[env_ids, vertex_ids, :3] = wall_nodal_state[env_ids, vertex_ids, :3]
        wall_nodal_kinematic_target[env_ids, vertex_ids, 3] = 0.0  # constrained
    soft_block.write_nodal_kinematic_target_to_sim(wall_nodal_kinematic_target)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Update existing buffers
    # Note: We need to update buffers before the first step for the controller.
    robot.update(dt=sim_dt)

    # Get the center of the robot soft joint limits
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)

    # get the updated states
    (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    ) = update_states(sim, robot, ee_frame_idx, arm_joint_ids, ft_body_idx)

    # Track the given target command
    current_goal_idx = 0  # Current goal index for the arm
    command = torch.zeros(
        scene.num_envs, osc.action_dim, device=sim.device
    )  # Generic target command, which can be pose, position, force, etc.
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the body frame
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the world frame (for marker)

    # Set joint efforts to zero
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)
    joint_efforts = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)
    finger_pos_des = torch.zeros(scene.num_envs, len(finger_joint_ids), device=sim.device)

    count = 0
    try:
        # Simulation loop
        while simulation_app.is_running():
            # reset every 500 steps
            if count % 500 == 0:
                # reset joint state to default
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone()
                # Keep gripper closed at all times.
                default_joint_pos[:, finger_joint_ids] = 0.0
                robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                robot.set_joint_effort_target(zero_joint_efforts)  # Set zero torques in the initial step
                robot.write_data_to_sim()
                robot.reset()

                # reset target pose
                robot.update(sim_dt)
                _, _, _, ee_pose_b, _, _, _, _, _, _ = update_states(
                    sim, robot, ee_frame_idx, arm_joint_ids, ft_body_idx
                )  # at reset, the jacobians are not updated to the latest state
                command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = update_target(
                    sim, scene, osc, root_pose_w, ee_target_set, current_goal_idx
                )
                # set the osc command
                osc.reset()
                command, task_frame_pose_b = convert_to_task_frame(
                    osc, command=command, ee_target_pose_b=ee_target_pose_b
                )
                osc.set_command(
                    command=command,
                    current_ee_pose_b=ee_pose_b,
                    current_task_frame_pose_b=task_frame_pose_b,
                )
                robot.set_joint_position_target(finger_pos_des, joint_ids=finger_joint_ids)
                robot.write_data_to_sim()
            else:
                # get the updated states
                (
                    jacobian_b,
                    mass_matrix,
                    gravity,
                    ee_pose_b,
                    ee_vel_b,
                    root_pose_w,
                    ee_pose_w,
                    ee_force_b,
                    joint_pos,
                    joint_vel,
                ) = update_states(sim, robot, ee_frame_idx, arm_joint_ids, ft_body_idx)
                # compute the joint commands
                joint_efforts = osc.compute(
                    jacobian_b=jacobian_b,
                    current_ee_pose_b=ee_pose_b,
                    current_ee_vel_b=ee_vel_b,
                    current_ee_force_b=ee_force_b,
                    mass_matrix=mass_matrix,
                    gravity=gravity,
                    current_joint_pos=joint_pos,
                    current_joint_vel=joint_vel,
                    nullspace_joint_pos_target=joint_centers,
                )
                # apply actions
                robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
                robot.set_joint_position_target(finger_pos_des, joint_ids=finger_joint_ids)
                robot.write_data_to_sim()

            soft_block.write_nodal_kinematic_target_to_sim(wall_nodal_kinematic_target)
            soft_block.write_data_to_sim()

            # update marker positions
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

            # perform step
            sim.step(render=True)
            # update robot buffers
            robot.update(sim_dt)
            # update buffers
            scene.update(sim_dt)
            observer_camera.update(sim_dt)

            # FT logging: env_0 incoming joint wrench at fr3_hand joint (child body fr3_hand).
            if ft_body_idx is not None:
                body_wrenches = robot.data.body_incoming_joint_wrench_b
                ft_value = body_wrenches[0, ft_body_idx].detach().cpu().tolist()
            else:
                ft_value = [None, None, None, None, None, None]
            ee_pos_b_env0 = ee_pose_b[0, 0:3].detach().cpu().tolist()
            ee_goal_pos_b_env0 = ee_target_pose_b[0, 0:3].detach().cpu().tolist()
            ee_pos_err_b_env0 = (ee_pose_b[0, 0:3] - ee_target_pose_b[0, 0:3]).detach().cpu().tolist()
            ee_pos_err_b_norm_env0 = float(torch.norm(ee_pose_b[0, 0:3] - ee_target_pose_b[0, 0:3]).item())

            if ft_writer is not None and ft_csv_file is not None:
                ft_writer.writerow(
                    [
                        datetime.now().isoformat(),
                        count,
                        args_cli.youngs_modulus,
                        ee_pos_b_env0[0],
                        ee_pos_b_env0[1],
                        ee_pos_b_env0[2],
                        ee_goal_pos_b_env0[0],
                        ee_goal_pos_b_env0[1],
                        ee_goal_pos_b_env0[2],
                        ee_pos_err_b_env0[0],
                        ee_pos_err_b_env0[1],
                        ee_pos_err_b_env0[2],
                        ee_pos_err_b_norm_env0,
                        "/World/envs/env_0/Robot/fr3/fr3_link8/fr3_hand_joint",
                        ft_value[0],
                        ft_value[1],
                        ft_value[2],
                        ft_value[3],
                        ft_value[4],
                        ft_value[5],
                    ]
                )
                ft_csv_file.flush()

            # Optional video recording from observer camera in env_0.
            if video_writer is not None:
                rgb_frame = observer_camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                if rgb_frame.dtype != "uint8":
                    rgb_frame = (rgb_frame * 255.0).clip(0, 255).astype("uint8")
                video_writer.append_data(rgb_frame)

                if args_cli.record_length > 0 and (count + 1) >= args_cli.record_length:
                    print(f"[INFO] Reached record length: {args_cli.record_length} steps.")
                    break

            # update sim-time
            count += 1
    finally:
        if ft_csv_file is not None:
            ft_csv_file.close()
        if video_writer is not None:
            video_writer.close()
            print(f"[INFO] Saved video: {video_path}")


# Update robot states
def update_states(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
    ft_body_idx: int | None,
):
    """Update the robot states.

    Args:
        sim: (SimulationContext) Simulation context.
        robot: (Articulation) Robot articulation.
        ee_frame_idx: (int) End-effector frame index.
        arm_joint_ids: (list[int]) Arm joint indices.
        ft_body_idx: (int | None) Body index for incoming joint wrench at fr3_hand.

    Returns:
        jacobian_b (torch.tensor): Jacobian in the body frame.
        mass_matrix (torch.tensor): Mass matrix.
        gravity (torch.tensor): Gravity vector.
        ee_pose_b (torch.tensor): End-effector pose in the body frame.
        ee_vel_b (torch.tensor): End-effector velocity in the body frame.
        root_pose_w (torch.tensor): Root pose in the world frame.
        ee_pose_w (torch.tensor): End-effector pose in the world frame.
        ee_force_b (torch.tensor): End-effector force in the body frame.
        joint_pos (torch.tensor): The joint positions.
        joint_vel (torch.tensor): The joint velocities.

    Raises:
        ValueError: Undefined target_type.
    """
    # obtain dynamics related quantities from simulation
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # Extract end-effector velocity in the world frame
    root_vel_w = robot.data.root_vel_w  # Extract root velocity in the world frame
    relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Use incoming joint wrench at fr3_hand joint as force feedback (body frame).
    if ft_body_idx is not None:
        ee_force_b = robot.data.body_incoming_joint_wrench_b[:, ft_body_idx, 0:3]
    else:
        ee_force_b = torch.zeros(robot.data.root_pos_w.shape[0], 3, device=sim.device)

    # Get joint positions and velocities
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]

    return (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    )


# Update the target commands
def update_target(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    osc: OperationalSpaceController,
    root_pose_w: torch.tensor,
    ee_target_set: torch.tensor,
    current_goal_idx: int,
):
    """Update the targets for the operational space controller.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        osc: (OperationalSpaceController) Operational space controller.
        root_pose_w: (torch.tensor) Root pose in the world frame.
        ee_target_set: (torch.tensor) End-effector target set.
        current_goal_idx: (int) Current goal index.

    Returns:
        command (torch.tensor): Updated target command.
        ee_target_pose_b (torch.tensor): Updated target pose in the body frame.
        ee_target_pose_w (torch.tensor): Updated target pose in the world frame.
        next_goal_idx (int): Next goal index.

    Raises:
        ValueError: Undefined target_type.
    """

    # update the ee desired command
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:] = ee_target_set[current_goal_idx]

    # update the ee desired pose
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            ee_target_pose_b[:] = command[:, :7]
        elif target_type == "wrench_abs":
            pass  # ee_target_pose_b could stay at the root frame for force control, what matters is ee_target_b
        else:
            raise ValueError("Undefined target_type within update_target().")

    # update the target desired pose in world frame (for marker)
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    next_goal_idx = (current_goal_idx + 1) % len(ee_target_set)

    return command, ee_target_pose_b, ee_target_pose_w, next_goal_idx


# Convert the target commands to the task frame
def convert_to_task_frame(osc: OperationalSpaceController, command: torch.tensor, ee_target_pose_b: torch.tensor):
    """Converts the target commands to the task frame.

    Args:
        osc: OperationalSpaceController object.
        command: Command to be converted.
        ee_target_pose_b: Target pose in the body frame.

    Returns:
        command (torch.tensor): Target command in the task frame.
        task_frame_pose_b (torch.tensor): Target pose in the task frame.

    Raises:
        ValueError: Undefined target_type.
    """
    command = command.clone()
    task_frame_pose_b = ee_target_pose_b.clone()

    cmd_idx = 0
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
            )
            cmd_idx += 7
        elif target_type == "wrench_abs":
            # These are already defined in target frame for ee_goal_wrench_set_tilted_task (since it is
            # easier), so not transforming
            cmd_idx += 6
        else:
            raise ValueError("Undefined target_type within _convert_to_task_frame().")

    return command, task_frame_pose_b


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
