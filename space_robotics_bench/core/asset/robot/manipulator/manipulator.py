from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Dict, Iterable, List, Mapping, Sequence, Tuple, Type

from space_robotics_bench.core.asset import ArticulationCfg
from space_robotics_bench.core.asset.robot.manipulator.manipulator_type import (
    ManipulatorType,
)
from space_robotics_bench.core.asset.robot.robot import Robot
from space_robotics_bench.core.asset.robot.robot_type import RobotType
from space_robotics_bench.utils import convert_to_snake_case


class Manipulator(Robot, robot_entrypoint=RobotType.MANIPULATOR):
    asset_cfg: ArticulationCfg

    def __init_subclass__(
        cls,
        manipulator_entrypoint: ManipulatorType | None = None,
        manipulator_metaclass: bool = False,
        robot_metaclass: bool = False,
        **kwargs,
    ):
        super().__init_subclass__(
            robot_metaclass=(
                robot_metaclass
                or manipulator_entrypoint is not None
                or manipulator_metaclass
            ),
            **kwargs,
        )
        if manipulator_entrypoint is not None:
            assert isinstance(
                manipulator_entrypoint, ManipulatorType
            ), f"Class '{cls.__name__}' is marked as a manipulator entrypoint, but '{manipulator_entrypoint}' is not a valid {ManipulatorType}"
            assert (
                manipulator_entrypoint not in ManipulatorRegistry.base_types.keys()
            ), f"Class '{cls.__name__}' is marked as '{manipulator_entrypoint}' manipulator entrypoint, but it was already marked by '{ManipulatorRegistry.base_types[manipulator_entrypoint].__name__}'"
            ManipulatorRegistry.base_types[manipulator_entrypoint] = cls
        elif manipulator_metaclass:
            ManipulatorRegistry.meta_types.append(cls)
        else:
            for manipulator_type, base in ManipulatorRegistry.base_types.items():
                if issubclass(cls, base):
                    if manipulator_type not in ManipulatorRegistry.registry.keys():
                        ManipulatorRegistry.registry[manipulator_type] = []
                    else:
                        assert (
                            convert_to_snake_case(cls.__name__)
                            not in (
                                convert_to_snake_case(manipulator.__name__)
                                for manipulator in ManipulatorRegistry.registry[
                                    manipulator_type
                                ]
                            )
                        ), f"Cannot register multiple manipulators with an identical name: '{cls.__module__}:{cls.__name__}' already exists as '{next(manipulator for manipulator in ManipulatorRegistry.registry[manipulator_type] if convert_to_snake_case(cls.__name__) == convert_to_snake_case(manipulator.__name__)).__module__}:{cls.__name__}'"
                    ManipulatorRegistry.registry[manipulator_type].append(cls)

    @cached_property
    def manipulator_type(self) -> ManipulatorType:
        for manipulator_type, base in ManipulatorRegistry.base_types.items():
            if isinstance(self, base):
                return manipulator_type
        raise ValueError(
            f"Class '{self.__class__.__name__}' has unknown manipulator type"
        )

    @classmethod
    def manipulator_registry(
        cls,
    ) -> Mapping[ManipulatorType, Sequence[Type[Manipulator]]]:
        return ManipulatorRegistry.registry

    @classmethod
    def robot_registry(cls) -> Sequence[Type[Manipulator]]:
        return super().robot_registry().get(RobotType.MANIPULATOR, [])  # type: ignore


class ManipulatorRegistry:
    registry: ClassVar[Dict[ManipulatorType, List[Type[Manipulator]]]] = {}
    base_types: ClassVar[Dict[ManipulatorType, Type[Manipulator]]] = {}
    meta_types: ClassVar[List[Type[Manipulator]]] = []

    @classmethod
    def keys(cls) -> Iterable[ManipulatorType]:
        return cls.registry.keys()

    @classmethod
    def items(cls) -> Iterable[Tuple[ManipulatorType, Iterable[Type[Manipulator]]]]:
        return cls.registry.items()

    @classmethod
    def values(cls) -> Iterable[Iterable[Type[Manipulator]]]:
        return cls.registry.values()

    @classmethod
    def values_inner(cls) -> Iterable[Type[Manipulator]]:
        return {
            manipulator
            for manipulators in cls.registry.values()
            for manipulator in manipulators
        }

    @classmethod
    def n_robots(cls) -> int:
        return sum(len(manipulators) for manipulators in cls.registry.values())

    @classmethod
    def registered_modules(cls) -> Iterable[str]:
        return {
            manipulator.__module__
            for manipulators in cls.registry.values()
            for manipulator in manipulators
        }

    @classmethod
    def registered_packages(cls) -> Iterable[str]:
        return {module.split(".", maxsplit=1)[0] for module in cls.registered_modules()}
