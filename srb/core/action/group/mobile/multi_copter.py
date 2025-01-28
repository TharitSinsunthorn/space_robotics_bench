from dataclasses import MISSING

import torch

from srb.core.action.action_group import ActionGroup
from srb.core.action.term import MultiCopterBodyVelocityActionCfg
from srb.utils.cfg import configclass


@configclass
class MultiCopterBodyVelocity(ActionGroup):
    vel: MultiCopterBodyVelocityActionCfg = MISSING  # type: ignore

    def map_commands(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        return torch.concat(
            (
                twist[:3],
                twist[5].unsqueeze(0),
            ),
        )
