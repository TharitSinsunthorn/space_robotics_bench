from __future__ import annotations

from typing import Sequence, Type

from space_robotics_bench.core.asset import (
    ArticulationCfg,
    AssetBaseCfg,
    FrameCfg,
    RigidObjectCfg,
)
from space_robotics_bench.core.asset.asset import Asset
from space_robotics_bench.core.asset.asset_type import AssetType


# TODO: Consider basing vehicle off a robot
class Vehicle(Asset, asset_entrypoint=AssetType.VEHICLE):
    ## Model
    asset_cfg: AssetBaseCfg | ArticulationCfg | RigidObjectCfg

    ## Frames
    frame_cargo_bay: FrameCfg
    frame_manipulator_base: FrameCfg
    frame_camera_base: FrameCfg | None

    @classmethod
    def asset_registry(cls) -> Sequence[Type[Vehicle]]:
        return super().asset_registry().get(AssetType.VEHICLE, [])  # type: ignore
