from __future__ import annotations

from typing import Sequence, Type

from space_robotics_bench.core.asset.asset import Asset
from space_robotics_bench.core.asset.asset_type import AssetType


class Terrain(Asset, asset_entrypoint=AssetType.TERRAIN):
    @classmethod
    def asset_registry(cls) -> Sequence[Type[Terrain]]:
        return super().asset_registry().get(AssetType.TERRAIN, [])  # type: ignore
