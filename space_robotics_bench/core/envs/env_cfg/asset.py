from enum import Enum, auto
from typing import Annotated

from pydantic import BaseModel

from space_robotics_bench.utils.typing import EnumNameSerializer


class AssetVariant(str, Enum):
    NONE = auto()
    PRIMITIVE = auto()
    DATASET = auto()
    PROCEDURAL = auto()

    def __str__(self) -> str:
        return self.name.lower()


class Asset(BaseModel):
    variant: Annotated[AssetVariant, EnumNameSerializer]


# TODO: Change defaults
class Assets(BaseModel):
    robot: Asset = Asset(variant=AssetVariant.DATASET)
    object: Asset = Asset(variant=AssetVariant.PROCEDURAL)
    terrain: Asset = Asset(variant=AssetVariant.PROCEDURAL)
    vehicle: Asset = Asset(variant=AssetVariant.DATASET)
