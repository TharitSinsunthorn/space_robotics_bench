import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from torch import pi

import srb.core.asset as asset_utils
import srb.utils.math as math_utils
from srb.core.actions import MultiCopterActionCfg, MultiCopterActionGroupCfg
from srb.core.asset import Frame, MultiCopter
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT


class Ingenuity(MultiCopter):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("ingenuity")
            .joinpath("ingenuity.usdc")
            .as_posix(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "rotors": ImplicitActuatorCfg(
                joint_names_expr=["rotor_joint_[1-2]"],
                velocity_limit=2500 / 60 * 2 * pi,
                effort_limit=7.5,
                stiffness=0.0,
                damping=1000.0,
            ),
        },
        soft_joint_pos_limit_factor=0.0,
    )

    ## Actions
    action_cfg: MultiCopterActionGroupCfg = MultiCopterActionGroupCfg(
        flight=MultiCopterActionCfg(
            asset_name="robot",
            frame_base="body",
            regex_joints_rotors="rotor_joint_[1-2]",
            nominal_rpm={
                "rotor_joint_1": 2500.0,
                "rotor_joint_2": -2500.0,
            },
            tilt_magnitude=0.125,
            scale=4.0,
        )
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="body")
    frame_camera_bottom: Frame = Frame(
        prim_relpath=f"{frame_base}/camera_bottom",
        offset=asset_utils.Transform(
            translation=(0.045, 0.0, 0.1275),
            rotation=math_utils.quat_from_rpy(0.0, 90.0, 0.0),
        ),
    )

    ## Links
    regex_links_rotors: str = "rotor_[1-2]"

    ## Joints
    regex_joints_rotors: str = "rotor_joint_[1-2]"
