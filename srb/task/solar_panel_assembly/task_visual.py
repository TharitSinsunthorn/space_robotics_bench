from typing import Dict

import torch

from srb.env import VisualManipulationEnvExt, VisualManipulationEnvExtCfg
from srb.utils import configclass

from .task import Task, TaskCfg


@configclass
class VisualTaskCfg(TaskCfg, VisualManipulationEnvExtCfg):
    def __post_init__(self):
        TaskCfg.__post_init__(self)
        VisualManipulationEnvExtCfg.__post_init__(self)


class VisualTask(Task, VisualManipulationEnvExt):
    cfg: VisualTaskCfg

    def __init__(self, cfg: VisualTaskCfg, **kwargs):
        Task.__init__(self, cfg, **kwargs)
        VisualManipulationEnvExt.__init__(self, cfg, **kwargs)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {
            **Task._get_observations(self),
            **VisualManipulationEnvExt._get_observations(self),
        }
