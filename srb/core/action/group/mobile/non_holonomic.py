from dataclasses import MISSING

import torch

from srb.core.action import NonHolonomicActionCfg
from srb.core.action.action_group import ActionGroup
from srb.utils import configclass


@configclass
class NonHolonomicDrive(ActionGroup):
    drive: NonHolonomicActionCfg = MISSING  # type: ignore

    def map_commands(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        return twist[:2]
