from __future__ import annotations

from typing import Sequence, Type

from srb.core.asset.robot.mobile_manipulation.mobile_manipulator import (
    MobileManipulator,
    MobileManipulatorRegistry,
)
from srb.core.asset.robot.mobile_manipulation.mobile_manipulator_type import (
    MobileManipulatorType,
)


class OrbitalManipulator(
    MobileManipulator, mobile_manipulator_entrypoint=MobileManipulatorType.ORBITAL
):
    @classmethod
    def mobile_manipulator_registry(cls) -> Sequence[Type[OrbitalManipulator]]:
        return MobileManipulatorRegistry.registry.get(MobileManipulatorType.ORBITAL, [])  # type: ignore
