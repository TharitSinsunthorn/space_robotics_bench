from srb.core.asset import AssetBaseCfg
from srb.core.asset.common import Frame
from srb.core.asset.object.object import Object


class StaticVehicle(Object, asset_metaclass=True):
    asset_cfg: AssetBaseCfg

    ## Frames
    frame_cargo_bay: Frame
    frame_manipulator_base: Frame
    frame_camera_base: Frame | None
