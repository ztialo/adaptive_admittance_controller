# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_3_CFG`: Franka Emika 3 robot with Panda hand
* :obj:`FRANKA_3_HIGH_PD_CFG`: Franka Emika 3 robot with Panda hand with stiffer PD control
* :obj:`FRANKA_ROBOTIQ_GRIPPER_CFG`: Franka robot with Robotiq_2f_85 gripper

Reference: https://github.com/frankaemika/franka_ros
"""
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FRANKA_USD_PATH = str(_REPO_ROOT / "assets" / "franka3.usd")

FRANKA_3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FRANKA_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": -0.569,
            "fr3_joint3": 0.0,
            "fr3_joint4": -2.810,
            "fr3_joint5": 0.0,
            "fr3_joint6": 3.037,
            "fr3_joint7": 0.741,
            "fr3_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "franka_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "franka_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "franka_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka 3 robot."""


FRANKA_3_HIGH_PD_CFG = FRANKA_3_CFG.copy()
FRANKA_3_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_3_HIGH_PD_CFG.actuators["franka_shoulder"].stiffness = 400.0
FRANKA_3_HIGH_PD_CFG.actuators["franka_shoulder"].damping = 80.0
FRANKA_3_HIGH_PD_CFG.actuators["franka_forearm"].stiffness = 400.0
FRANKA_3_HIGH_PD_CFG.actuators["franka_forearm"].damping = 80.0
"""Configuration of Franka Emika robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""


FRANKA_ROBOTIQ_GRIPPER_CFG = FRANKA_3_CFG.copy()
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.usd_path = _FRANKA_USD_PATH
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2F_85"}
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -0.569,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.810,
    "fr3_joint5": 0.0,
    "fr3_joint6": 3.037,
    "fr3_joint7": 0.741,
    "finger_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_knuckle_joint": 0.0,
    ".*_outer_.*_joint": 0.0,
}
FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.pos = (-0.85, 0, 0.76)
FRANKA_ROBOTIQ_GRIPPER_CFG.actuators = {
    "franka_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint[1-4]"],
        effort_limit_sim=5200.0,
        velocity_limit_sim=2.175,
        stiffness=1100.0,
        damping=80.0,
    ),
    "franka_forearm": ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint[5-7]"],
        effort_limit_sim=720.0,
        velocity_limit_sim=2.61,
        stiffness=1000.0,
        damping=80.0,
    ),
    "gripper_drive": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint
        effort_limit_sim=1650,
        velocity_limit_sim=10.0,
        stiffness=17,
        damping=0.02,
    ),
    # enable the gripper to grasp in a parallel manner
    "gripper_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=50,
        velocity_limit_sim=10.0,
        stiffness=0.2,
        damping=0.001,
    ),
    # set PD to zero for passive joints in close-loop gripper
    "gripper_passive": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
        effort_limit_sim=1.0,
        velocity_limit_sim=10.0,
        stiffness=0.0,
        damping=0.0,
    ),
}


"""Configuration of Franka Emika Panda robot with Robotiq_2f_85 gripper."""
