from srb.core.asset import Articulation
from srb.core.envs import DirectEnv

from .cfg import BaseAerialRoboticsEnvCfg


class BaseAerialRoboticsEnv(DirectEnv):
    cfg: BaseAerialRoboticsEnvCfg

    def __init__(self, cfg: BaseAerialRoboticsEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get handles to scene assets
        self._robot: Articulation = self.scene["robot"]
