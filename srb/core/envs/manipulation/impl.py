from srb.core.asset import Articulation
from srb.core.envs import DirectEnv
from srb.core.sensors import ContactSensor, FrameTransformer

from .cfg import BaseManipulationEnvCfg


class BaseManipulationEnv(DirectEnv):
    cfg: BaseManipulationEnvCfg

    def __init__(self, cfg: BaseManipulationEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get handles to scene assets
        self._robot: Articulation = self.scene["robot"]
        self._tf_robot_ee: FrameTransformer = self.scene["tf_robot_ee"]
        self._contacts_robot: ContactSensor = self.scene["contacts_robot"]
