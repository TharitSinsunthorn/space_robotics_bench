from omni.isaac.lab.sim import UsdFileCfg
from torch import pi

from srb.core.action import SingleArmIK_BinaryGripper
from srb.core.actuator import ImplicitActuatorCfg
from srb.core.asset import ArticulationCfg, Frame, SingleArmManipulator, Transform
from srb.core.controller import DifferentialIKControllerCfg
from srb.core.mdp import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from srb.core.sim import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg
from srb.utils.math import quat_from_rpy
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT


class Canadarm3Large(SingleArmManipulator):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("canadarm3_large")
            .joinpath("canadarm3_large.usdc")
            .as_posix(),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "canadarm3_large_joint_1": 50.0 * pi / 180.0,
                "canadarm3_large_joint_2": 0.0 * pi / 180.0,
                "canadarm3_large_joint_3": 55.0 * pi / 180.0,
                "canadarm3_large_joint_4": 75.0 * pi / 180.0,
                "canadarm3_large_joint_5": -30.0 * pi / 180.0,
                "canadarm3_large_joint_6": 0.0 * pi / 180.0,
                "canadarm3_large_joint_7": 0.0 * pi / 180.0,
            },
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=["canadarm3_large_joint_[1-7]"],
                effort_limit=2500.0,
                velocity_limit=5.0,
                stiffness=40000.0,
                damping=25000.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    ## Actions
    action_cfg: SingleArmIK_BinaryGripper = SingleArmIK_BinaryGripper(
        arm=DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["canadarm3_large_joint_[1-7]"],
            body_name="canadarm3_large_7",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.1,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0)
            ),
        ),
        hand=BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["canadarm3_large_joint_7"],
            close_command_expr={"canadarm3_large_joint_7": 0.0},
            open_command_expr={"canadarm3_large_joint_7": 0.0},
        ),
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="canadarm3_large_0")
    frame_ee: Frame = Frame(
        prim_relpath="canadarm3_large_7",
        offset=Transform(translation=(0.0, 0.0, 0.1034)),
    )
    frame_camera_base: Frame = Frame(
        prim_relpath="canadarm3_large_0/camera_base",
        offset=Transform(
            translation=(0.06, 0.0, 0.15),
            rotation=quat_from_rpy(0.0, -10.0, 0.0),
        ),
    )
    frame_camera_wrist: Frame = Frame(
        prim_relpath="canadarm3_large_7/camera_wrist",
        offset=Transform(
            translation=(0.0, 0.0, -0.45),
            rotation=quat_from_rpy(0.0, 90.0, 180.0),
        ),
    )

    ## Links
    regex_links_arm: str = "canadarm3_large_[0-6]"
    regex_links_hand: str = "canadarm3_large_7"

    ## Joints
    regex_joints_arm: str = "canadarm3_large_joint_[1-7]"
    regex_joints_hand: str = "canadarm3_large_joint_7"
