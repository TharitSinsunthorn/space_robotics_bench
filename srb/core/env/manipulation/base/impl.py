from srb.core.asset import Articulation
from srb.core.env import DirectEnv
from srb.core.sensor import ContactSensor, FrameTransformer

from .cfg import ManipulationEnvCfg


class ManipulationEnv(DirectEnv):
    cfg: ManipulationEnvCfg

    def __init__(self, cfg: ManipulationEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get handles to scene assets
        self._robot: Articulation = self.scene["robot"]
        self._tf_robot_ee: FrameTransformer = self.scene["tf_robot_ee"]
        self._contacts_robot: ContactSensor = self.scene["contacts_robot"]
