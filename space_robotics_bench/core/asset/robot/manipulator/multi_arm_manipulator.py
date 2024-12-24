from __future__ import annotations

from typing import Sequence, Type

from space_robotics_bench.core.asset.robot.manipulator.manipulator import Manipulator
from space_robotics_bench.core.asset.robot.manipulator.manipulator_type import (
    ManipulatorType,
)


class MultiArmManipulator(
    Manipulator, manipulator_entrypoint=ManipulatorType.MULTI_ARM
):
    @classmethod
    def manipulator_registry(cls) -> Sequence[Type[MultiArmManipulator]]:
        return super().manipulator_registry().get(ManipulatorType.MULTI_ARM, [])  # type: ignore
