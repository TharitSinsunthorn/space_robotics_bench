from omni.isaac.lab.sensors import ContactSensor

from srb.core.asset import Articulation
from srb.core.envs import BaseEnv

from .cfg import BaseLocomotionEnvCfg


class BaseLocomotionEnv(BaseEnv):
    cfg: BaseLocomotionEnvCfg

    def __init__(self, cfg: BaseLocomotionEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get handles to scene assets
        self._robot: Articulation = self.scene["robot"]
        self._contacts_robot: ContactSensor = self.scene["contacts_robot"]
        # self._height_scanner: RayCaster = self.scene["height_scanner"]
