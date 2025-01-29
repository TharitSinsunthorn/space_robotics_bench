from __future__ import annotations

from enum import Enum, auto


class InterfaceType(str, Enum):
    GUI = auto()
    ROS2 = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, string: str) -> InterfaceType:
        try:
            return next(variant for variant in cls if string.upper() == variant.name)
        except StopIteration:
            raise ValueError(f'String "{string}" is not a valid "{cls.__name__}"')
