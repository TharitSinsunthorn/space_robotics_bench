from srb.core.asset import LeggedRobot
from srb.core.env.common.extension.visual import VisualExtCfg
from srb.core.sensor import CameraCfg, PinholeCameraCfg
from srb.utils.cfg import configclass
from srb.utils.math import quat_from_rpy

from .cfg import LocomotionEnvCfg


@configclass
class LocomotionEnvVisualExtCfg(LocomotionEnvCfg, VisualExtCfg):
    def __post_init__(self):
        LocomotionEnvCfg.__post_init__(self)
        assert isinstance(self.robot, LeggedRobot)

        self.cameras_cfg = {
            "cam_scene": CameraCfg(
                prim_path=f"{self.scene.robot.prim_path}{('/' + self.robot.frame_base.prim_relpath) if self.robot.frame_base.prim_relpath else ''}/camera_scene",
                offset=CameraCfg.OffsetCfg(
                    convention="world",
                    pos=(-2.5, 0.0, 2.5),
                    rot=quat_from_rpy(0.0, 45.0, 0.0),
                ),
                spawn=PinholeCameraCfg(
                    clipping_range=(0.05, 75.0 - 0.05),
                ),
            ),
        }

        VisualExtCfg.__post_init__(self)
