from dataclasses import MISSING

import torch

from srb.core.action import (
    BinaryJointPositionActionCfg,
    BinaryJointVelocityActionCfg,
    EMAJointPositionToLimitsActionCfg,
    JointEffortActionCfg,
    JointPositionActionCfg,
    JointPositionToLimitsActionCfg,
    JointVelocityActionCfg,
    RelativeJointPositionActionCfg,
)
from srb.core.action.action_group import ActionGroup
from srb.utils import configclass


@configclass
class JointPosition(ActionGroup):
    pos: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"]
    )


@configclass
class JointPositionRelative(ActionGroup):
    pos: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"]
    )


@configclass
class JointPositionBounded(ActionGroup):
    pos: JointPositionToLimitsActionCfg = JointPositionToLimitsActionCfg(
        asset_name="robot", joint_names=[".*"]
    )


@configclass
class JointPositionBoundedEMA(ActionGroup):
    pos: EMAJointPositionToLimitsActionCfg = EMAJointPositionToLimitsActionCfg(
        asset_name="robot", joint_names=[".*"]
    )


@configclass
class JointPositionBinary(ActionGroup):
    binary: BinaryJointPositionActionCfg = MISSING  # type: ignore

    def map_commands(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        return torch.Tensor((-1.0 if event else 1.0,)).to(device=twist.device)


@configclass
class JointVelocity(ActionGroup):
    vel: JointVelocityActionCfg = JointVelocityActionCfg(
        asset_name="robot", joint_names=[".*"]
    )

    def map_commands(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        return torch.Tensor((-1.0 if event else 1.0,)).to(device=twist.device)


@configclass
class JointVelocityBinary(ActionGroup):
    binary: BinaryJointVelocityActionCfg = MISSING  # type: ignore


@configclass
class JointEffort(ActionGroup):
    eff: JointEffortActionCfg = JointEffortActionCfg(
        asset_name="robot", joint_names=[".*"]
    )
