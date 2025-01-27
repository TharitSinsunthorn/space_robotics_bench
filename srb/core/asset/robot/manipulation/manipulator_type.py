from enum import Enum, auto


class ManipulatorType(str, Enum):
    MULTI_ARM = auto()
    SINGLE_ARM = auto()

    def __str__(self) -> str:
        return self.name.lower()
