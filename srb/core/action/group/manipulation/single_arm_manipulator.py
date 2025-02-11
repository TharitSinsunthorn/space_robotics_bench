from dataclasses import MISSING

import torch

from srb.core.action import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    OperationalSpaceControllerActionCfg,
)
from srb.core.action.action_group import ActionGroup
from srb.utils.cfg import configclass


@configclass
class SingleArmIK(ActionGroup):
    arm: DifferentialInverseKinematicsActionCfg = MISSING  # type: ignore

    def map_commands(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        return twist


@configclass
class SingleArmIK_BinaryGripper(ActionGroup):
    arm: DifferentialInverseKinematicsActionCfg = MISSING  # type: ignore
    hand: BinaryJointPositionActionCfg = MISSING  # type: ignore

    def map_commands(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        return torch.concat(
            (twist, torch.Tensor((-1.0 if event else 1.0,)).to(device=twist.device))
        )


@configclass
class SingleArmOSC(ActionGroup):
    arm: OperationalSpaceControllerActionCfg = MISSING  # type: ignore

    def map_commands(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        # TODO: Map teleop actions based on the impedance mode of the OSC
        raise NotImplementedError


@configclass
class SingleArmOSC_BinaryGripper(ActionGroup):
    arm: OperationalSpaceControllerActionCfg = MISSING  # type: ignore
    hand: BinaryJointPositionActionCfg = MISSING  # type: ignore

    def map_commands(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        # TODO: Map teleop actions based on the impedance mode of the OSC
        raise NotImplementedError
