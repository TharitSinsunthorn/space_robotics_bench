from srb.core.asset import Articulation
from srb.core.env import DirectEnv
from srb.core.sensor import ContactSensor

from .cfg import LocomotionEnvCfg


class LocomotionEnv(DirectEnv):
    cfg: LocomotionEnvCfg

    def __init__(self, cfg: LocomotionEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get handles to scene assets
        self._robot: Articulation = self.scene["robot"]
        self._contacts_robot: ContactSensor = self.scene["contacts_robot"]
