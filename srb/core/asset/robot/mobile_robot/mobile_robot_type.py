from enum import Enum, auto


class MobileRobotType(str, Enum):
    AERIAL = auto()
    GROUND = auto()
    SPACECRAFT = auto()

    def __str__(self) -> str:
        return self.name.lower()
