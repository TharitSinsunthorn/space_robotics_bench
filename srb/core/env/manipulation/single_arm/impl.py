from srb.core.asset import Articulation
from srb.core.env import DirectEnv
from srb.core.sensor import ContactSensor, FrameTransformer

from .cfg import SingleArmEnvCfg


class SingleArmEnv(DirectEnv):
    cfg: SingleArmEnvCfg

    def __init__(self, cfg: SingleArmEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._robot: Articulation = self.scene["robot"]
        self._tf_robot_ee: FrameTransformer = self.scene["tf_robot_ee"]
        self._contacts_robot_arm: ContactSensor = self.scene["contacts_robot_arm"]
