from typing import Dict

import torch

from srb.core.env import LocomotionEnvVisualExtCfg, VisualExt
from srb.utils.cfg import configclass

from .task import Task, TaskCfg


@configclass
class VisualTaskCfg(TaskCfg, LocomotionEnvVisualExtCfg):
    def __post_init__(self):
        TaskCfg.__post_init__(self)
        LocomotionEnvVisualExtCfg.__post_init__(self)


class VisualTask(Task, VisualExt):
    cfg: VisualTaskCfg

    def __init__(self, cfg: VisualTaskCfg, **kwargs):
        Task.__init__(self, cfg, **kwargs)
        VisualExt.__init__(self, cfg, **kwargs)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {
            **Task._get_observations(self),
            **VisualExt._get_observations(self),
        }
