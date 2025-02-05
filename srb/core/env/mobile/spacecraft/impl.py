from srb.core.asset import Articulation
from srb.core.env import DirectEnv

from .cfg import SpacecraftEnvCfg


class SpacecraftEnv(DirectEnv):
    cfg: SpacecraftEnvCfg

    def __init__(self, cfg: SpacecraftEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._robot: Articulation = self.scene["robot"]
