from srb.core.asset import Articulation
from srb.core.env import DirectEnv

from .cfg import AerialEnvCfg


class AerialEnv(DirectEnv):
    cfg: AerialEnvCfg

    def __init__(self, cfg: AerialEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get handles to scene assets
        self._robot: Articulation = self.scene["robot"]
