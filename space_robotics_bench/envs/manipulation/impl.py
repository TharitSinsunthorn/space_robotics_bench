from omni.isaac.lab.sensors import ContactSensor, FrameTransformer

from space_robotics_bench.core.assets import Articulation
from space_robotics_bench.core.envs import BaseEnv

from .cfg import BaseManipulationEnvCfg


class BaseManipulationEnv(BaseEnv):
    cfg: BaseManipulationEnvCfg

    def __init__(self, cfg: BaseManipulationEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get handles to scene assets
        self._robot: Articulation = self.scene["robot"]
        self._tf_robot_ee: FrameTransformer = self.scene["tf_robot_ee"]
        self._contacts_robot: ContactSensor = self.scene["contacts_robot"]
