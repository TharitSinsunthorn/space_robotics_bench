from enum import Enum, auto


class MobileManipulatorType(str, Enum):
    AERIAL = auto()
    GROUND = auto()
    HUMANOID = auto()
    ORBITAL = auto()

    def __str__(self) -> str:
        return self.name.lower()
