from __future__ import annotations

from typing import Sequence, Tuple, Type

from pydantic import PositiveFloat

from space_robotics_bench.core.asset.asset import Asset
from space_robotics_bench.core.asset.asset_type import AssetType


class Terrain(Asset, asset_entrypoint=AssetType.TERRAIN):
    # TODO: Rename to size
    scale: Tuple[PositiveFloat, PositiveFloat, PositiveFloat] | None = None
    density: PositiveFloat | None = None

    @classmethod
    def asset_registry(cls) -> Sequence[Type[Terrain]]:
        return super().asset_registry().get(AssetType.TERRAIN, [])  # type: ignore


class Surface(Terrain, asset_metaclass=True):
    flat_area_size: PositiveFloat | None = None


class Underground(Terrain, asset_metaclass=True):
    pass


class Interior(Terrain, asset_metaclass=True):
    pass


class Exterior(Terrain, asset_metaclass=True):
    pass
