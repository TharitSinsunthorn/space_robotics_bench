from srb.core.env.common.extension.visual import VisualExtCfg
from srb.core.sensor import CameraCfg, PinholeCameraCfg
from srb.utils import configclass
from srb.utils.math import quat_from_rpy

from .cfg import ManipulationEnvCfg


@configclass
class ManipulationEnvVisualExtCfg(ManipulationEnvCfg, VisualExtCfg):
    rerender_on_reset: bool = True

    def __post_init__(self):
        ManipulationEnvCfg.__post_init__(self)

        self.cameras_cfg = {
            "cam_scene": CameraCfg(
                prim_path="{ENV_REGEX_NS}/camera_scene",
                offset=CameraCfg.OffsetCfg(
                    convention="world",
                    pos=(1.2, 0.0, 0.8),
                    rot=quat_from_rpy(0.0, 30.0, 180.0),
                ),
                spawn=PinholeCameraCfg(
                    clipping_range=(0.01, 4.0 - 0.01),
                ),
            ),
            "cam_base": CameraCfg(
                prim_path=f"{self.scene.robot.prim_path}/{self.robot.frame_camera_base.prim_relpath}",
                offset=CameraCfg.OffsetCfg(
                    convention="world",
                    pos=self.robot.frame_camera_base.offset.translation,
                    rot=self.robot.frame_camera_base.offset.rotation,
                ),
                spawn=PinholeCameraCfg(
                    focal_length=5.0,
                    horizontal_aperture=12.0,
                    clipping_range=(0.001, 2.5 - 0.001),
                ),
            ),
            "cam_wrist": CameraCfg(
                prim_path=f"{self.scene.robot.prim_path}/{self.robot.frame_camera_wrist.prim_relpath}",
                offset=CameraCfg.OffsetCfg(
                    convention="world",
                    pos=self.robot.frame_camera_wrist.offset.translation,
                    rot=self.robot.frame_camera_wrist.offset.rotation,
                ),
                spawn=PinholeCameraCfg(
                    focal_length=10.0,
                    horizontal_aperture=16.0,
                    clipping_range=(0.001, 1.5 - 0.001),
                ),
            ),
        }

        if self.vehicle and self.vehicle.frame_camera_base:
            cam = self.cameras_cfg["cam_base"]
            cam.prim_path = f"{self.scene.vehicle.prim_path}/{self.vehicle.frame_camera_base.prim_relpath}"
            cam.offset = CameraCfg.OffsetCfg(
                convention="world",
                pos=self.vehicle.frame_camera_base.offset.translation,
                rot=self.vehicle.frame_camera_base.offset.rotation,
            )

        VisualExtCfg.__post_init__(self)
