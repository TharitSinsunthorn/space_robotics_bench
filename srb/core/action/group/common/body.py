import torch

from srb.core.action.action_group import ActionGroup
from srb.core.action.term import BodyVelocityActionCfg
from srb.utils.cfg import configclass


@configclass
class BodyVelocity(ActionGroup):
    vel: BodyVelocityActionCfg = BodyVelocityActionCfg(asset_name="robot")

    def map_commands(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        return twist
