from srb.core.env.common.extension.visual import VisualExtCfg
from srb.core.sensor import CameraCfg, PinholeCameraCfg
from srb.utils import configclass
from srb.utils.math import quat_from_rpy

from .cfg import AerialEnvCfg


@configclass
class AerialEnvVisualExtCfg(AerialEnvCfg, VisualExtCfg):
    rerender_on_reset: bool = True

    def __post_init__(self):
        AerialEnvCfg.__post_init__(self)

        self.cameras_cfg = {
            "cam_scene": CameraCfg(
                prim_path=f"{self.scene.robot.prim_path}/{self.robot.frame_base.prim_relpath}/camera_scene",
                offset=CameraCfg.OffsetCfg(
                    convention="world",
                    pos=(-2.5, 0.0, 2.5),
                    rot=quat_from_rpy(0.0, 45.0, 0.0),
                ),
                spawn=PinholeCameraCfg(
                    clipping_range=(0.05, 75.0 - 0.05),
                ),
            ),
            "cam_bottom": CameraCfg(
                prim_path=f"{self.scene.robot.prim_path}/{self.robot.frame_camera_bottom.prim_relpath}",
                offset=CameraCfg.OffsetCfg(
                    convention="world",
                    pos=self.robot.frame_camera_bottom.offset.translation,
                    rot=self.robot.frame_camera_bottom.offset.rotation,
                ),
                spawn=PinholeCameraCfg(
                    focal_length=5.0,
                    horizontal_aperture=12.0,
                    clipping_range=(0.05, 50.0 - 0.05),
                ),
            ),
        }

        VisualExtCfg.__post_init__(self)
