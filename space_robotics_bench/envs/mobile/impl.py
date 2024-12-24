from space_robotics_bench.core.asset import Articulation
from space_robotics_bench.core.envs import BaseEnv

from .cfg import BaseMobileRoboticsEnvCfg


class BaseMobileRoboticsEnv(BaseEnv):
    cfg: BaseMobileRoboticsEnvCfg

    def __init__(self, cfg: BaseMobileRoboticsEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get handles to scene assets
        self._robot: Articulation = self.scene["robot"]
