from enum import Enum, auto


class AssetVariant(str, Enum):
    PRIMITIVE = auto()
    DATASET = auto()
    PROCEDURAL = auto()

    def __str__(self) -> str:
        return self.name.lower()
