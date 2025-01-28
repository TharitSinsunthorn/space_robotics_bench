from dataclasses import MISSING

import torch

from srb.core.action.action_group import ActionGroup
from srb.core.action.term import WheeledRoverDriveActionCfg
from srb.utils import configclass


@configclass
class WheeledRoverDrive(ActionGroup):
    drive: WheeledRoverDriveActionCfg = MISSING  # type: ignore

    def map_commands(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        return twist[:2]
