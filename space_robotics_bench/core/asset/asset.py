from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Dict, Iterable, List, Mapping, Sequence, Tuple, Type

from pydantic import BaseModel, InstanceOf

from space_robotics_bench.core.asset import AssetBaseCfg
from space_robotics_bench.core.asset.asset_type import AssetType
from space_robotics_bench.utils import convert_to_snake_case


class Asset(BaseModel):
    ## Model
    asset_cfg: InstanceOf[AssetBaseCfg]

    @cached_property
    def name(self) -> str:
        return convert_to_snake_case(self.__class__.__name__)

    @cached_property
    def is_randomizable(self) -> bool:
        return False

    def __new__(cls, *args, **kwargs):
        if cls in (
            Asset,
            *AssetRegistry.base_types.keys(),
            *AssetRegistry.meta_types,
        ):
            raise TypeError(f"Cannot instantiate abstract class {cls.__name__}")
        return super().__new__(cls)

    def __init_subclass__(
        cls,
        asset_entrypoint: AssetType | None = None,
        asset_metaclass: bool = False,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        if asset_entrypoint is not None:
            assert isinstance(
                asset_entrypoint, AssetType
            ), f"Class '{cls.__name__}' is marked as an asset entrypoint, but '{asset_entrypoint}' is not a valid {AssetType}"
            assert (
                asset_entrypoint not in AssetRegistry.base_types.keys()
            ), f"Class '{cls.__name__}' is marked as '{asset_entrypoint}' asset entrypoint, but it was already marked by '{AssetRegistry.base_types[asset_entrypoint].__name__}'"
            AssetRegistry.base_types[asset_entrypoint] = cls
        elif asset_metaclass:
            AssetRegistry.meta_types.append(cls)
        else:
            for asset_type, base in AssetRegistry.base_types.items():
                if issubclass(cls, base):
                    if asset_type not in AssetRegistry.registry.keys():
                        AssetRegistry.registry[asset_type] = []
                    AssetRegistry.registry[asset_type].append(cls)

    @cached_property
    def asset_type(self) -> AssetType:
        for asset_type, base in AssetRegistry.base_types.items():
            if isinstance(self, base):
                return asset_type
        raise ValueError(f"Class '{self.__class__.__name__}' has unknown asset type")

    @classmethod
    def asset_registry(cls) -> Mapping[AssetType, Sequence[Type[Asset]]]:
        return AssetRegistry.registry


class AssetRegistry:
    registry: ClassVar[Dict[AssetType, List[Type[Asset]]]] = {}
    base_types: ClassVar[Dict[AssetType, Type[Asset]]] = {}
    meta_types: ClassVar[List[Type[Asset]]] = []

    @classmethod
    def keys(cls) -> Iterable[AssetType]:
        return cls.registry.keys()

    @classmethod
    def items(cls) -> Iterable[Tuple[AssetType, Type[Asset]]]:
        return cls.registry.items()

    @classmethod
    def values(cls) -> Iterable[Sequence[Type[Asset]]]:
        return cls.registry.values()

    @classmethod
    def values_inner(cls) -> Iterable[Type[Asset]]:
        return {asset for assets in cls.registry.values() for asset in assets}

    @classmethod
    def n_assets(cls) -> int:
        return sum(len(assets) for assets in cls.registry.values())

    @classmethod
    def registered_modules(cls) -> Iterable[str]:
        return {
            asset.__module__ for assets in cls.registry.values() for asset in assets
        }

    @classmethod
    def registered_packages(cls) -> Iterable[str]:
        return {module.split(".", maxsplit=1)[0] for module in cls.registered_modules()}
