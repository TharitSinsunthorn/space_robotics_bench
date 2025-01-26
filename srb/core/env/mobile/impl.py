from srb.core.asset import Articulation
from srb.core.env import DirectEnv

from .cfg import BaseMobileRoboticsEnvCfg


class BaseMobileRoboticsEnv(DirectEnv):
    cfg: BaseMobileRoboticsEnvCfg

    def __init__(self, cfg: BaseMobileRoboticsEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get handles to scene assets
        self._robot: Articulation = self.scene["robot"]
