from __future__ import annotations

from typing import Sequence, Type

from srb.core.asset.robot.mobile_manipulator.mobile_manipulator import (
    MobileManipulator,
    MobileManipulatorRegistry,
)
from srb.core.asset.robot.mobile_manipulator.mobile_manipulator_type import (
    MobileManipulatorType,
)


class GroundManipulator(
    MobileManipulator, mobile_manipulator_entrypoint=MobileManipulatorType.GROUND
):
    @classmethod
    def mobile_manipulator_registry(cls) -> Sequence[Type[GroundManipulator]]:
        return MobileManipulatorRegistry.registry.get(MobileManipulatorType.GROUND, [])  # type: ignore
