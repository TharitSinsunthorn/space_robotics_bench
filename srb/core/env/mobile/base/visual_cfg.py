from srb.core.env.common.extension.visual import VisualExtCfg
from srb.core.sensor import CameraCfg, PinholeCameraCfg
from srb.utils.cfg import configclass
from srb.utils.math import quat_from_rpy

from .cfg import MobileRoboticsEnvCfg


@configclass
class MobileRoboticsEnvVisualExtCfg(MobileRoboticsEnvCfg, VisualExtCfg):
    rerender_on_reset: bool = True

    def __post_init__(self):
        MobileRoboticsEnvCfg.__post_init__(self)

        self.cameras_cfg = {
            "cam_scene": CameraCfg(
                prim_path=f"{self.scene.robot.prim_path}/{self.robot.frame_base.prim_relpath}/camera_scene",
                offset=CameraCfg.OffsetCfg(
                    convention="world",
                    pos=(0.0, 7.5, 5.0),
                    rot=quat_from_rpy(0.0, 30.0, -90.0),
                ),
                spawn=PinholeCameraCfg(
                    clipping_range=(0.01, 20.0 - 0.01),
                ),
            ),
            "cam_front": CameraCfg(
                prim_path=f"{self.scene.robot.prim_path}/{self.robot.frame_camera_front.prim_relpath}",
                offset=CameraCfg.OffsetCfg(
                    convention="world",
                    pos=self.robot.frame_camera_front.offset.translation,
                    rot=self.robot.frame_camera_front.offset.rotation,
                ),
                spawn=PinholeCameraCfg(
                    focal_length=5.0,
                    horizontal_aperture=12.0,
                    clipping_range=(0.01, 20.0 - 0.01),
                ),
            ),
        }

        VisualExtCfg.__post_init__(self)
