from dataclasses import MISSING

import torch

from srb.core.actions import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from srb.core.actions.action_group import ActionGroup
from srb.utils import configclass


@configclass
class ManipulatorTaskSpaceActionCfg(ActionGroup):
    arm: DifferentialInverseKinematicsActionCfg = MISSING
    hand: BinaryJointPositionActionCfg = MISSING

    def map_teleop_actions(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        return torch.concat(
            (twist, torch.Tensor((-1.0 if event else 1.0,)).to(device=twist.device))
        )
