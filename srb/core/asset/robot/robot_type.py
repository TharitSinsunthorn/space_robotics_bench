from enum import Enum, auto


class RobotType(str, Enum):
    MANIPULATOR = auto()
    MOBILE_MANIPULATOR = auto()
    MOBILE_ROBOT = auto()

    def __str__(self) -> str:
        return self.name.lower()
