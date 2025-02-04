from srb.core.asset import WheeledRobot
from srb.core.env.common.extension.visual import VisualExtCfg
from srb.core.sensor import CameraCfg, PinholeCameraCfg
from srb.utils.cfg import configclass
from srb.utils.math import quat_from_rpy

from .cfg import WheeledEnvCfg


@configclass
class WheeledEnvVisualExtCfg(VisualExtCfg):
    def wrap(self, env_cfg: WheeledEnvCfg):
        assert isinstance(env_cfg.robot, WheeledRobot)

        self.cameras_cfg = {
            "cam_scene": CameraCfg(
                prim_path=f"{env_cfg.scene.robot.prim_path}{('/' + env_cfg.robot.frame_base.prim_relpath) if env_cfg.robot.frame_base.prim_relpath else ''}/camera_scene",
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
                prim_path=f"{env_cfg.scene.robot.prim_path}/{env_cfg.robot.frame_camera_front.prim_relpath}",
                offset=CameraCfg.OffsetCfg(
                    convention="world",
                    pos=env_cfg.robot.frame_camera_front.offset.translation,
                    rot=env_cfg.robot.frame_camera_front.offset.rotation,
                ),
                spawn=PinholeCameraCfg(
                    focal_length=5.0,
                    horizontal_aperture=12.0,
                    clipping_range=(0.01, 20.0 - 0.01),
                ),
            ),
        }

        super().wrap(env_cfg)
