import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

import srb.core.asset as asset_utils
import srb.utils.math as math_utils
from srb.core.actions import WheeledRoverActionCfg, WheeledRoverActionGroupCfg
from srb.core.asset import Frame, WheeledRobot
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT


class Perseverance(WheeledRobot):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("perseverance")
            .joinpath("perseverance.usdc")
            .as_posix(),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02, rest_offset=0.005
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.5,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "drive_joints": ImplicitActuatorCfg(
                joint_names_expr=["drive_joint.*"],
                velocity_limit=40.0,
                effort_limit=150.0,
                damping=25000.0,
                stiffness=0.0,
            ),
            "steer_joints": ImplicitActuatorCfg(
                joint_names_expr=["steer_joint.*"],
                velocity_limit=2.0,
                effort_limit=400.0,
                damping=200.0,
                stiffness=500.0,
            ),
            "rocker_joints": ImplicitActuatorCfg(
                joint_names_expr=["suspension_joint_rocker.*"],
                velocity_limit=5.0,
                effort_limit=2500.0,
                damping=400.0,
                stiffness=4000.0,
            ),
            "bogie_joints": ImplicitActuatorCfg(
                joint_names_expr=["suspension_joint_bogie.*"],
                velocity_limit=4.0,
                effort_limit=500.0,
                damping=25.0,
                stiffness=200.0,
            ),
        },
        # soft_joint_pos_limit_factor=0.0,
    )

    ## Actions
    action_cfg: WheeledRoverActionGroupCfg = WheeledRoverActionGroupCfg(
        WheeledRoverActionCfg(
            asset_name="robot",
            wheelbase=(2.26, 2.14764),
            wheelbase_mid=2.39164,
            wheel_radius=0.26268,
            steering_joint_names=[
                "steer_joint_front_left",
                "steer_joint_front_right",
                "steer_joint_rear_left",
                "steer_joint_rear_right",
            ],
            drive_joint_names=[
                "drive_joint_front_left",
                "drive_joint_front_right",
                "drive_joint_rear_left",
                "drive_joint_rear_right",
                "drive_joint_mid_left",
                "drive_joint_mid_right",
            ],
            scale=1.0,
        )
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="body")
    frame_camera_front: Frame = Frame(
        prim_relpath="body/camera_front",
        offset=asset_utils.Transform(
            # translation=(-0.3437, -0.8537, 1.9793),  # Left Navcam
            translation=(-0.7675, -0.8537, 1.9793),  # Right Navcam
            rotation=math_utils.quat_from_rpy(0.0, 15.0, -90.0),
        ),
    )

    ## Joints
    regex_drive_joints: str = "drive_joint.*"
    regex_steer_joints: str = "steer_joint.*"
