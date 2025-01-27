from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from srb.core.action.action_group import ActionGroup
from srb.core.asset import Articulation
from srb.core.manager import ActionTerm, ActionTermCfg
from srb.utils import configclass

if TYPE_CHECKING:
    from srb.core.env import DirectEnv


class SpacecraftAction(ActionTerm):
    cfg: "SpacecraftActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "SpacecraftActionCfg", env: "DirectEnv"):
        super().__init__(cfg, env)

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

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass


@configclass
class SpacecraftActionCfg(ActionTermCfg):
    class_type: ActionTerm = SpacecraftAction
    scale: float = 1.0


@configclass
class SpacecraftActionGroupCfg(ActionGroup):
    flight: SpacecraftActionCfg = MISSING

    def map_teleop_actions(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        return twist
