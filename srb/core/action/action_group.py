from __future__ import annotations

import torch

from srb.utils import configclass

# TODO: Add registry (so that it can be overridden as env.robot.action="another_registered_action_group")


@configclass
class ActionGroup:
    def map_teleop_actions(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        raise NotImplementedError()

    def supports_policy_teleop(self) -> bool:
        return False
