from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab_assets import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG

from srb.core.actions import JointPositionActionCfg, LocomotionJointSpaceActionCfg
from srb.core.asset import Frame, LeggedRobot


class AnymalB(LeggedRobot):
    ## Model
    asset_cfg: ArticulationCfg = ANYMAL_B_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

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
    asset_cfg: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

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
    asset_cfg: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    ## Actions
    action_cfg: LocomotionJointSpaceActionCfg = LocomotionJointSpaceActionCfg(
        joint_pos=JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="base")
