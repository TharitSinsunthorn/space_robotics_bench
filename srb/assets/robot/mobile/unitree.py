from isaaclab.sim import UsdFileCfg

from srb.core.action import JointPosition, JointPositionActionCfg
from srb.core.actuator import ActuatorNetMLPCfg
from srb.core.asset import ArticulationCfg, Frame, LeggedRobot
from srb.core.sim import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg
from srb.utils.nucleus import ISAACLAB_NUCLEUS_DIR, get_local_or_nucleus_path
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT

GO1_ACTUATOR_CFG = ActuatorNetMLPCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    network_file=get_local_or_nucleus_path(
        SRB_ASSETS_DIR_SRB_ROBOT.joinpath("Unitree")
        .joinpath("Go1")
        .joinpath("unitree_go1.pt"),
        f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/Unitree/unitree_go1.pt",
    ),
    pos_scale=-1.0,
    vel_scale=1.0,
    torque_scale=1.0,
    input_order="pos_vel",
    input_idx=[0, 1, 2],
    effort_limit=23.7,
    velocity_limit=30.0,
    saturation_effort=23.7,
)


class UnitreeGo1(LeggedRobot):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=UsdFileCfg(
            usd_path=get_local_or_nucleus_path(
                SRB_ASSETS_DIR_SRB_ROBOT.joinpath("Unitree")
                .joinpath("Go1")
                .joinpath("go1.usd"),
                f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd",
            ),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "base_legs": GO1_ACTUATOR_CFG,
        },
    )

    ## Actions
    action_cfg: JointPosition = JointPosition(
        pos=JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5)
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="base")

    ## Links
    regex_links_feet: str = ".*foot"
    regex_links_undesired_contacts: str = "(trunk|.*hip|.*thigh)"
