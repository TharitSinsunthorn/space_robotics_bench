from __future__ import annotations

import torch
from omni.isaac.lab.utils import configclass


@configclass
class ActionGroup:
    def map_teleop_actions(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        raise NotImplementedError()

    def supports_policy_teleop(self) -> bool:
        return False
