from isaaclab.sim import UsdFileCfg
from torch import pi

from srb.core.action import SingleArmIK_BinaryGripper
from srb.core.actuator import ImplicitActuatorCfg
from srb.core.asset import ArticulationCfg, Frame, SingleArmManipulator, Transform
from srb.core.controller import DifferentialIKControllerCfg
from srb.core.mdp import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from srb.core.sim import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from srb.utils.math import quat_from_rpy
from srb.utils.nucleus import ISAACLAB_NUCLEUS_DIR, get_local_or_nucleus_path
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT


class Franka(SingleArmManipulator):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=UsdFileCfg(
            usd_path=get_local_or_nucleus_path(
                SRB_ASSETS_DIR_SRB_ROBOT.joinpath("FrankaEmika").joinpath(
                    "panda_instanceable.usd"
                ),
                f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            ),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -(pi / 8.0),
                "panda_joint3": 0.0,
                "panda_joint4": -(pi - (pi / 8.0)),
                "panda_joint5": 0.0,
                "panda_joint6": pi - (pi / 4.0),
                "panda_joint7": (pi / 4.0),
                "panda_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=4000.0,
                damping=800.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=4000.0,
                damping=800.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    ## Actions
    action_cfg: SingleArmIK_BinaryGripper = SingleArmIK_BinaryGripper(
        arm=DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.1,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.107)
            ),
        ),
        hand=BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint.*"],
            close_command_expr={"panda_finger_joint.*": 0.0},
            open_command_expr={"panda_finger_joint.*": 0.04},
        ),
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="panda_link0")
    frame_ee: Frame = Frame(
        prim_relpath="panda_hand",
        offset=Transform(
            translation=(0.0, 0.0, 0.1034),
        ),
    )
    frame_camera_base: Frame = Frame(
        prim_relpath="panda_link0/camera_base",
        offset=Transform(
            translation=(0.06, 0.0, 0.15),
            rotation=quat_from_rpy(0.0, -10.0, 0.0),
        ),
    )
    frame_camera_wrist: Frame = Frame(
        prim_relpath="panda_hand/camera_wrist",
        offset=Transform(
            translation=(0.07, 0.0, 0.05),
            rotation=quat_from_rpy(0.0, -60.0, 180.0),
        ),
    )

    ## Links
    regex_links_arm: str = "panda_link[1-7]"
    regex_links_hand: str = "panda_(hand|leftfinger|rightfinger)"

    ## Joints
    regex_joints_arm: str = "panda_joint.*"
    regex_joints_hand: str = "panda_finger_joint.*"
