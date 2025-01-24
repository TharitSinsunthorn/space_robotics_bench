from typing import Dict

import torch

from srb.env import VisualAerialRoboticsEnvExt, VisualAerialRoboticsEnvExtCfg
from srb.utils import configclass

from .task import Task, TaskCfg


@configclass
class VisualTaskCfg(TaskCfg, VisualAerialRoboticsEnvExtCfg):
    def __post_init__(self):
        TaskCfg.__post_init__(self)
        VisualAerialRoboticsEnvExtCfg.__post_init__(self)


class VisualTask(Task, VisualAerialRoboticsEnvExt):
    cfg: VisualTaskCfg

    def __init__(self, cfg: VisualTaskCfg, **kwargs):
        Task.__init__(self, cfg, **kwargs)
        VisualAerialRoboticsEnvExt.__init__(self, cfg, **kwargs)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {
            **Task._get_observations(self),
            **VisualAerialRoboticsEnvExt._get_observations(self),
        }
