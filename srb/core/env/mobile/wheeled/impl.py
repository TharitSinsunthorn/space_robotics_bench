from srb.core.asset import Articulation
from srb.core.env import DirectEnv

from .cfg import WheeledEnvCfg


class WheeledEnv(DirectEnv):
    cfg: WheeledEnvCfg

    def __init__(self, cfg: WheeledEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get handles to scene assets
        self._robot: Articulation = self.scene["robot"]
