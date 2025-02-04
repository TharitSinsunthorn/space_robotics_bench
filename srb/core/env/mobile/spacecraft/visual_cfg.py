from srb.core.asset import Spacecraft
from srb.core.env.common.extension.visual import VisualExtCfg
from srb.core.sensor import CameraCfg, PinholeCameraCfg
from srb.utils.cfg import configclass
from srb.utils.math import quat_from_rpy

from .cfg import SpacecraftEnvCfg


@configclass
class SpacecraftEnvVisualExtCfg(VisualExtCfg):
    def wrap(self, env_cfg: SpacecraftEnvCfg):
        assert isinstance(env_cfg.robot, Spacecraft)

        self.cameras_cfg = {
            "cam_scene": CameraCfg(
                prim_path=f"{env_cfg.scene.robot.prim_path}{('/' + env_cfg.robot.frame_base.prim_relpath) if env_cfg.robot.frame_base.prim_relpath else ''}/camera_scene",
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

        super().wrap(env_cfg)
