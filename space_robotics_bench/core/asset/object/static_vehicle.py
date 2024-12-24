from space_robotics_bench.core.asset import AssetBaseCfg
from space_robotics_bench.core.asset.common import Frame
from space_robotics_bench.core.asset.object.object import Object


class StaticVehicle(Object, asset_metaclass=True):
    asset_cfg: AssetBaseCfg

    ## Frames
    frame_cargo_bay: Frame
    frame_manipulator_base: Frame
    frame_camera_base: Frame | None
