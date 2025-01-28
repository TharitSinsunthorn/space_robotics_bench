from typing import Dict

import torch

from srb.core.env import ManipulationEnvVisualExtCfg, VisualExt
from srb.utils.cfg import configclass

from .task import Task, TaskCfg
from .task_multi import MultiTask, MultiTaskCfg


@configclass
class VisualTaskCfg(TaskCfg, ManipulationEnvVisualExtCfg):
    def __post_init__(self):
        TaskCfg.__post_init__(self)
        ManipulationEnvVisualExtCfg.__post_init__(self)


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


@configclass
class MultiVisualTaskCfg(MultiTaskCfg, ManipulationEnvVisualExtCfg):
    def __post_init__(self):
        MultiTaskCfg.__post_init__(self)
        ManipulationEnvVisualExtCfg.__post_init__(self)


class MultiVisualTask(MultiTask, VisualExt):
    cfg: MultiVisualTaskCfg

    def __init__(self, cfg: MultiVisualTaskCfg, **kwargs):
        MultiTask.__init__(self, cfg, **kwargs)
        VisualExt.__init__(self, cfg, **kwargs)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {
            **MultiTask._get_observations(self),
            **VisualExt._get_observations(self),
        }
