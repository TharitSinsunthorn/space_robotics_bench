from omni.isaac.lab.sim import UsdFileCfg
from omni.isaac.lab_assets import ANYDRIVE_3_LSTM_ACTUATOR_CFG

from srb.core.actions import JointPositionActionCfg, LocomotionJointSpaceActionCfg
from srb.core.asset import ArticulationCfg, Frame, LeggedRobot
from srb.core.sim import (
    ArticulationRootPropertiesCfg,
    MultiAssetSpawnerCfg,
    RigidBodyPropertiesCfg,
)
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT


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
    action_cfg: LocomotionJointSpaceActionCfg = LocomotionJointSpaceActionCfg(
        joint_pos=JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
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
    action_cfg: LocomotionJointSpaceActionCfg = LocomotionJointSpaceActionCfg(
        joint_pos=JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
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
    action_cfg: LocomotionJointSpaceActionCfg = LocomotionJointSpaceActionCfg(
        joint_pos=JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
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
    action_cfg: LocomotionJointSpaceActionCfg = LocomotionJointSpaceActionCfg(
        joint_pos=JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="base")
