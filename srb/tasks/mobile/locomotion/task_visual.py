from typing import Dict

import torch

from srb.core.env import LocomotionEnvVisualExtCfg, VisualExt
from srb.utils.cfg import configclass

from .task import Task, TaskCfg


@configclass
class VisualTaskCfg(LocomotionEnvVisualExtCfg, TaskCfg):
    ## Visualization
    visualize_commands: bool = False

    def __post_init__(self):
        TaskCfg.__post_init__(self)
        LocomotionEnvVisualExtCfg.wrap(self, env_cfg=self)


class VisualTask(VisualExt, Task):
    cfg: VisualTaskCfg

    def __init__(self, cfg: VisualTaskCfg, **kwargs):
        Task.__init__(self, cfg, **kwargs)
        VisualExt.__init__(self, cfg, **kwargs)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {
            **Task._get_observations(self),
            **VisualExt._get_observations(self),
        }
