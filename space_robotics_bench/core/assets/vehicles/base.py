from space_robotics_bench.core.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    AssetCfg,
    FrameCfg,
    RigidObjectCfg,
)


class VehicleCfg(AssetCfg):
    ## Model
    asset_cfg: AssetBaseCfg | ArticulationCfg | RigidObjectCfg

    ## Frames
    frame_manipulator_base: FrameCfg
    frame_camera_base: FrameCfg | None
    frame_cargo_bay: FrameCfg
