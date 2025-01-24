from srb.core.asset import Articulation
from srb.core.envs import DirectEnv

from .cfg import BaseSpacecraftRoboticsEnvCfg


class BaseSpacecraftRoboticsEnv(DirectEnv):
    cfg: BaseSpacecraftRoboticsEnvCfg

    def __init__(self, cfg: BaseSpacecraftRoboticsEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get handles to scene assets
        self._robot: Articulation = self.scene["robot"]
