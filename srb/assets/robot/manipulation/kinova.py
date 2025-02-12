from isaaclab.sim import UsdFileCfg
from torch import pi

from srb.core.action import SingleArmIK
from srb.core.actuator import ImplicitActuatorCfg
from srb.core.asset import ArticulationCfg, Frame, SingleArmManipulator, Transform
from srb.core.controller import DifferentialIKControllerCfg
from srb.core.mdp import DifferentialInverseKinematicsActionCfg
from srb.core.sim import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from srb.utils.math import quat_from_rpy
from srb.utils.nucleus import ISAAC_NUCLEUS_DIR, get_local_or_nucleus_path
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT


class KinovaGen3(SingleArmManipulator):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=UsdFileCfg(
            usd_path=get_local_or_nucleus_path(
                SRB_ASSETS_DIR_SRB_ROBOT.joinpath("Kinova")
                .joinpath("Gen3")
                .joinpath("gen3n7_instanceable.usd"),
                f"{ISAAC_NUCLEUS_DIR}/Robots/Kinova/Gen3/gen3n7_instanceable.usd",
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
                "joint_1": 0.0,
                "joint_2": pi / 3.0,
                "joint_3": 0.0,
                "joint_4": pi / 3.0,
                "joint_5": 0.0,
                "joint_6": pi / 4.0,
                "joint_7": 0.0,
            },
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=4000.0,
                damping=800.0,
            ),
            "forearm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=4000.0,
                damping=800.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    ## Actions
    action_cfg: SingleArmIK = SingleArmIK(
        arm=DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_.*"],
            body_name="bracelet_link",
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
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="base_link")
    frame_ee: Frame = Frame(
        prim_relpath="bracelet_link",
        offset=Transform(
            translation=(0.0, 0.0, 0.1034),
        ),
    )
    frame_camera_base: Frame = Frame(
        prim_relpath="base_link/camera_base",
        offset=Transform(
            translation=(0.06, 0.0, 0.15),
            rotation=quat_from_rpy(0.0, -10.0, 0.0),
        ),
    )
    frame_camera_wrist: Frame = Frame(
        prim_relpath="bracelet_link/camera_wrist",
        offset=Transform(
            translation=(0.07, 0.0, 0.05),
            rotation=quat_from_rpy(0.0, -60.0, 180.0),
        ),
    )

    ## Links
    regex_links_arm: str = "(shoulder|half_arm_1|half_arm_2|forearm|spherical_wrist_1|spherical_wrist_2|)_link"
    regex_links_hand: str = "bracelet_link"

    ## Joints
    regex_joints_arm: str = "joint_.*"
    regex_joints_hand: str | None = None
