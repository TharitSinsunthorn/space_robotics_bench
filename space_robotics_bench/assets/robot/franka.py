from os import path

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp import DifferentialInverseKinematicsActionCfg
from torch import pi

import space_robotics_bench.core.asset as asset_utils
import space_robotics_bench.utils.math as math_utils
from space_robotics_bench.core.actions import ManipulatorTaskSpaceActionCfg
from space_robotics_bench.core.envs import mdp
from space_robotics_bench.paths import SRB_ASSETS_DIR_SRB_ROBOT


def franka_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/robot",
    asset_name: str = "robot",
    use_relative_mode: bool = True,
    action_scale: float = 0.1,
    **kwargs,
) -> asset_utils.ManipulatorCfg:
    frame_base = "panda_link0"
    frame_ee = "panda_hand"
    regex_joints_arm = "panda_joint.*"
    regex_joints_hand = "panda_finger_joint.*"

    return asset_utils.ManipulatorCfg(
        ## Model
        asset_cfg=ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=path.join(
                    SRB_ASSETS_DIR_SRB_ROBOT,
                    "franka",
                    "panda.usdc",
                ),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=1,
                ),
                # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
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
        ).replace(prim_path=prim_path, **kwargs),
        ## Actions
        action_cfg=ManipulatorTaskSpaceActionCfg(
            arm=mdp.DifferentialInverseKinematicsActionCfg(
                asset_name=asset_name,
                joint_names=[regex_joints_arm],
                body_name=frame_ee,
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=use_relative_mode,
                    ik_method="dls",
                ),
                scale=action_scale,
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.107)
                ),
            ),
            hand=mdp.BinaryJointPositionActionCfg(
                asset_name=asset_name,
                joint_names=[regex_joints_hand],
                close_command_expr={regex_joints_hand: 0.0},
                open_command_expr={regex_joints_hand: 0.04},
            ),
        ),
        ## Frames
        base_frame=asset_utils.FrameCfg(
            prim_relpath=frame_base,
        ),
        frame_ee=asset_utils.FrameCfg(
            prim_relpath=frame_ee,
            offset=asset_utils.TransformCfg(
                translation=(0.0, 0.0, 0.1034),
            ),
        ),
        frame_camera_base=asset_utils.FrameCfg(
            prim_relpath=f"{frame_base}/camera_base",
            offset=asset_utils.TransformCfg(
                translation=(0.06, 0.0, 0.15),
                rotation=math_utils.quat_from_rpy(0.0, -10.0, 0.0),
            ),
        ),
        frame_camera_wrist=asset_utils.FrameCfg(
            prim_relpath=f"{frame_ee}/camera_wrist",
            offset=asset_utils.TransformCfg(
                translation=(0.07, 0.0, 0.05),
                rotation=math_utils.quat_from_rpy(0.0, -60.0, 180.0),
            ),
        ),
        ## Links
        regex_links_arm="panda_link[1-7]",
        regex_links_hand="panda_(hand|leftfinger|rightfinger)",
        ## Joints
        regex_joints_arm=regex_joints_arm,
        regex_joints_hand=regex_joints_hand,
    )
