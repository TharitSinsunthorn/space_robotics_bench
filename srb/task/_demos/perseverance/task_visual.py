from typing import Dict

import torch

from srb.core.envs import VisualMobileRoboticsEnvExt, VisualMobileRoboticsEnvExtCfg
from srb.utils import configclass

from .task import Task, TaskCfg


@configclass
class VisualTaskCfg(TaskCfg, VisualMobileRoboticsEnvExtCfg):
    def __post_init__(self):
        TaskCfg.__post_init__(self)
        VisualMobileRoboticsEnvExtCfg.__post_init__(self)


class VisualTask(Task, VisualMobileRoboticsEnvExt):
    cfg: VisualTaskCfg

    def __init__(self, cfg: VisualTaskCfg, **kwargs):
        Task.__init__(self, cfg, **kwargs)
        VisualMobileRoboticsEnvExt.__init__(self, cfg, **kwargs)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {
            **Task._get_observations(self),
            **VisualMobileRoboticsEnvExt._get_observations(self),
        }
