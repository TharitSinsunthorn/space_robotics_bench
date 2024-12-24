from enum import Enum, auto


class AssetType(str, Enum):
    OBJECT = auto()
    ROBOT = auto()
    TERRAIN = auto()

    def __str__(self) -> str:
        return self.name.lower()
