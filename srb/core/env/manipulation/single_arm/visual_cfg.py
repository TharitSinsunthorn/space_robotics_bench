from srb.core.asset import SingleArmManipulator, StaticVehicle
from srb.core.env.common.extension.visual import VisualExtCfg
from srb.core.sensor import CameraCfg, PinholeCameraCfg
from srb.utils.cfg import configclass
from srb.utils.math import quat_from_rpy

from .cfg import SingleArmEnvCfg


@configclass
class SingleArmEnvVisualExtCfg(VisualExtCfg):
    def wrap(self, env_cfg: SingleArmEnvCfg):
        assert isinstance(env_cfg.robot, SingleArmManipulator)

        self.cameras_cfg = {
            "cam_scene": CameraCfg(
                prim_path=f"{env_cfg.scene.robot.prim_path}{('/' + env_cfg.robot.frame_base.prim_relpath) if env_cfg.robot.frame_base.prim_relpath else ''}/camera_scene",
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
                prim_path=f"{env_cfg.scene.robot.prim_path}/{env_cfg.robot.frame_camera_base.prim_relpath}",
                offset=CameraCfg.OffsetCfg(
                    convention="world",
                    pos=env_cfg.robot.frame_camera_base.offset.translation,
                    rot=env_cfg.robot.frame_camera_base.offset.rotation,
                ),
                spawn=PinholeCameraCfg(
                    focal_length=5.0,
                    horizontal_aperture=12.0,
                    clipping_range=(0.001, 2.5 - 0.001),
                ),
            ),
            "cam_wrist": CameraCfg(
                prim_path=f"{env_cfg.scene.robot.prim_path}/{env_cfg.robot.frame_camera_wrist.prim_relpath}",
                offset=CameraCfg.OffsetCfg(
                    convention="world",
                    pos=env_cfg.robot.frame_camera_wrist.offset.translation,
                    rot=env_cfg.robot.frame_camera_wrist.offset.rotation,
                ),
                spawn=PinholeCameraCfg(
                    focal_length=10.0,
                    horizontal_aperture=16.0,
                    clipping_range=(0.001, 1.5 - 0.001),
                ),
            ),
        }

        if (
            isinstance(env_cfg.vehicle, StaticVehicle)
            and env_cfg.vehicle.frame_camera_base
        ):
            assert env_cfg.scene.vehicle is not None
            self.cameras_cfg["cam_base"].offset = CameraCfg.OffsetCfg(
                convention="world",
                pos=env_cfg.vehicle.frame_camera_base.offset.translation,
                rot=env_cfg.vehicle.frame_camera_base.offset.rotation,
            )

        super().wrap(env_cfg)
