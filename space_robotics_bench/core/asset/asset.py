from __future__ import annotations

from functools import cached_property
from itertools import chain
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Type,
)

from pydantic import BaseModel
from simforge import BlGeometry, BlModel
from simforge.integrations.isaaclab import SimforgeAssetCfg
from simforge.typing import TexResConfig

from space_robotics_bench.core.asset import AssetBaseCfg
from space_robotics_bench.core.asset.asset_type import AssetType
from space_robotics_bench.utils import convert_to_snake_case, logging


class Asset(BaseModel):
    INPUT_BLOCKLIST: ClassVar[Set] = {"asset_cfg"}
    asset_cfg: AssetBaseCfg

    texture_resolution: TexResConfig | None = None

    @cached_property
    def name(self) -> str:
        return convert_to_snake_case(self.__class__.__name__)

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
                    else:
                        assert (
                            convert_to_snake_case(cls.__name__)
                            not in (
                                convert_to_snake_case(asset.__name__)
                                for asset in AssetRegistry.registry[asset_type]
                            )
                        ), f"Cannot register multiple assets with an identical name: '{cls.__module__}:{cls.__name__}' already exists as '{next(asset for asset in AssetRegistry.registry[asset_type] if convert_to_snake_case(cls.__name__) == convert_to_snake_case(asset.__name__)).__module__}:{cls.__name__}'"
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

    @property
    def inputs(self) -> Mapping[str, Any]:
        return {
            k: v
            for k, v in {
                k: getattr(self, k)
                for k in chain(
                    self.__class__.model_fields.keys(),
                    self.model_computed_fields.keys(),
                )
                if k not in self.INPUT_BLOCKLIST
            }.items()
            if v is not None
        }

    def model_post_init(self, __context):
        super().model_post_init(__context)

        # Apply direct attributes as inputs to the underlying asset configuration
        inputs = self.inputs
        if len(inputs) == 0:
            return
        if isinstance(self.asset_cfg.spawn, SimforgeAssetCfg):
            for asset in self.asset_cfg.spawn.assets:
                match asset:
                    case geo if isinstance(geo, BlGeometry):
                        for k, v in inputs.items():
                            for op in geo.ops:
                                if hasattr(op, k):
                                    setattr(op, k, v)
                    case model if isinstance(model, BlModel):
                        for k, v in inputs.items():
                            for op in model.geo.ops:
                                if hasattr(op, k):
                                    setattr(op, k, v)
                            if model.mat is not None and hasattr(model.mat.shader, k):
                                setattr(model.mat.shader, k, v)
                            if hasattr(model, k):
                                setattr(model, k, v)
                    case _:
                        logging.warning(
                            f"SimForge asset of type '{type(asset)}' is not supported for input processing"
                        )
        else:
            for k, v in inputs.items():
                if hasattr(self.asset_cfg.spawn, k):
                    setattr(self.asset_cfg.spawn, k, v)


class AssetRegistry:
    registry: ClassVar[Dict[AssetType, List[Type[Asset]]]] = {}
    base_types: ClassVar[Dict[AssetType, Type[Asset]]] = {}
    meta_types: ClassVar[List[Type[Asset]]] = []

    @classmethod
    def keys(cls) -> Iterable[AssetType]:
        return cls.registry.keys()

    @classmethod
    def items(cls) -> Iterable[Tuple[AssetType, Iterable[Type[Asset]]]]:
        return cls.registry.items()

    @classmethod
    def values(cls) -> Iterable[Iterable[Type[Asset]]]:
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
