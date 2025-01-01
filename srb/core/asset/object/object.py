from __future__ import annotations

from typing import Sequence, Type

from srb.core.asset import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from srb.core.asset.asset import Asset
from srb.core.asset.asset_type import AssetType


class Object(Asset, asset_entrypoint=AssetType.OBJECT):
    asset_cfg: AssetBaseCfg | ArticulationCfg | RigidObjectCfg

    @classmethod
    def asset_registry(cls) -> Sequence[Type[Object]]:
        return super().asset_registry().get(AssetType.OBJECT, [])  # type: ignore
