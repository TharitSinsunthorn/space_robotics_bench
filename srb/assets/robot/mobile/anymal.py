from isaaclab.sim import UsdFileCfg

from srb.core.action import JointPosition, JointPositionActionCfg
from srb.core.actuator import ActuatorNetLSTMCfg, DCMotorCfg
from srb.core.asset import ArticulationCfg, Frame, LeggedRobot
from srb.core.sim import (
    ArticulationRootPropertiesCfg,
    MultiAssetSpawnerCfg,
    RigidBodyPropertiesCfg,
)
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT

ANYDRIVE_3_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
)

ANYDRIVE_3_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    network_file=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("anymal")
    .joinpath("anydrive_3_lstm_jit.pt")
    .as_posix(),
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
)


class AnymalB(LeggedRobot):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("anymal")
            .joinpath("anymal_b")
            .joinpath("anymal_b.usd")
            .as_posix(),
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
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*HAA": 0.0,  # all HAA
                ".*F_HFE": 0.4,  # both front HFE
                ".*H_HFE": -0.4,  # both hind HFE
                ".*F_KFE": -0.8,  # both front KFE
                ".*H_KFE": 0.8,  # both hind KFE
            },
        ),
        actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
        soft_joint_pos_limit_factor=0.95,
    )

    ## Actions
    action_cfg: JointPosition = JointPosition(
        pos=JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5)
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="base")


class AnymalC(LeggedRobot):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("anymal")
            .joinpath("anymal_c")
            .joinpath("anymal_c.usd")
            .as_posix(),
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
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*HAA": 0.0,  # all HAA
                ".*F_HFE": 0.4,  # both front HFE
                ".*H_HFE": -0.4,  # both hind HFE
                ".*F_KFE": -0.8,  # both front KFE
                ".*H_KFE": 0.8,  # both hind KFE
            },
        ),
        actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
        soft_joint_pos_limit_factor=0.95,
    )

    ## Actions
    action_cfg: JointPosition = JointPosition(
        pos=JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5)
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="base")


class AnymalD(LeggedRobot):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("anymal")
            .joinpath("anymal_d")
            .joinpath("anymal_d.usd")
            .as_posix(),
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
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*HAA": 0.0,  # all HAA
                ".*F_HFE": 0.4,  # both front HFE
                ".*H_HFE": -0.4,  # both hind HFE
                ".*F_KFE": -0.8,  # both front KFE
                ".*H_KFE": 0.8,  # both hind KFE
            },
        ),
        actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
        soft_joint_pos_limit_factor=0.95,
    )

    ## Actions
    action_cfg: JointPosition = JointPosition(
        pos=JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5)
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="base")


class AnymalMulti(LeggedRobot):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=MultiAssetSpawnerCfg(
            random_choice=False,
            assets_cfg=[
                # NOTE: AnymalB seems to be incompatible with AnymalC and AnymalD
                # UsdFileCfg(
                #     usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("anymal")
                #     .joinpath("anymal_b")
                #     .joinpath("anymal_b.usd")
                #     .as_posix(),
                #     activate_contact_sensors=True,
                #     rigid_props=RigidBodyPropertiesCfg(
                #         disable_gravity=False,
                #         retain_accelerations=False,
                #         linear_damping=0.0,
                #         angular_damping=0.0,
                #         max_linear_velocity=1000.0,
                #         max_angular_velocity=1000.0,
                #         max_depenetration_velocity=1.0,
                #     ),
                #     articulation_props=ArticulationRootPropertiesCfg(
                #         enabled_self_collisions=True,
                #         solver_position_iteration_count=4,
                #         solver_velocity_iteration_count=0,
                #     ),
                # ),
                UsdFileCfg(
                    usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("anymal")
                    .joinpath("anymal_c")
                    .joinpath("anymal_c.usd")
                    .as_posix(),
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
                        enabled_self_collisions=True,
                        solver_position_iteration_count=4,
                        solver_velocity_iteration_count=0,
                    ),
                ),
                UsdFileCfg(
                    usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("anymal")
                    .joinpath("anymal_d")
                    .joinpath("anymal_d.usd")
                    .as_posix(),
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
                        enabled_self_collisions=True,
                        solver_position_iteration_count=4,
                        solver_velocity_iteration_count=0,
                    ),
                ),
            ],
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
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*HAA": 0.0,  # all HAA
                ".*F_HFE": 0.4,  # both front HFE
                ".*H_HFE": -0.4,  # both hind HFE
                ".*F_KFE": -0.8,  # both front KFE
                ".*H_KFE": 0.8,  # both hind KFE
            },
        ),
        actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
        soft_joint_pos_limit_factor=0.95,
    )

    ## Actions
    action_cfg: JointPosition = JointPosition(
        pos=JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5)
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="base")
