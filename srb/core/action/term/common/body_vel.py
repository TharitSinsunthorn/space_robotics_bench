from typing import TYPE_CHECKING, Type

import torch

from srb.core.manager import ActionTerm, ActionTermCfg
from srb.utils import configclass

if TYPE_CHECKING:
    from srb.core.asset import Articulation, RigidObject


class BodyVelAction(ActionTerm):
    cfg: "BodyVelocityActionCfg"
    _asset: "Articulation | RigidObject"

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions = actions
        self._processed_actions = self.raw_actions * self.cfg.scale

    def apply_actions(self):
        current_velocity = self._asset._data.body_vel_w[:, 0].squeeze(1)

        applied_velocities = current_velocity + self.processed_actions
        self._asset.write_root_velocity_to_sim(applied_velocities)


@configclass
class BodyVelocityActionCfg(ActionTermCfg):
    class_type: Type = BodyVelAction

    scale: float = 1.0
