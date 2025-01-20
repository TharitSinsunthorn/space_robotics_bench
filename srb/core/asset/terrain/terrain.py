from __future__ import annotations

from typing import Sequence, Type

from pydantic import PositiveFloat

from srb.core.asset import AssetBaseCfg
from srb.core.asset.asset import Asset, AssetRegistry
from srb.core.asset.asset_type import AssetType


class Terrain(Asset, asset_entrypoint=AssetType.TERRAIN):
    asset_cfg: AssetBaseCfg
    density: PositiveFloat | None = None

    @classmethod
    def asset_registry(cls) -> Sequence[Type[Terrain]]:
        return AssetRegistry.registry.get(AssetType.TERRAIN, [])  # type: ignore


class Surface(Terrain, asset_metaclass=True):
    flat_area_size: PositiveFloat | None = None


class Underground(Terrain, asset_metaclass=True):
    pass


class Interior(Terrain, asset_metaclass=True):
    pass


class Exterior(Terrain, asset_metaclass=True):
    pass
