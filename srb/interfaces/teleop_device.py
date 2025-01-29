from __future__ import annotations

from enum import Enum, auto


class TeleopDevice(str, Enum):
    GAMEPAD = auto()
    HAPTIC = auto()
    KEYBOARD = auto()
    ROS2 = auto()
    SPACEMOUSE = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, string: str) -> TeleopDevice:
        try:
            return next(variant for variant in cls if string.upper() == variant.name)
        except StopIteration:
            raise ValueError(f'String "{string}" is not a valid "{cls.__name__}"')
