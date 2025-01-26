from __future__ import annotations

from typing import Sequence, Type

from srb.core.action import SpacecraftActionGroupCfg
from srb.core.asset.robot.mobile_robot.mobile_robot import (
    MobileRobot,
    MobileRobotRegistry,
)
from srb.core.asset.robot.mobile_robot.mobile_robot_type import MobileRobotType


class Spacecraft(
    MobileRobot,
    mobile_robot_entrypoint=MobileRobotType.SPACECRAFT,
    arbitrary_types_allowed=True,
):
    action_cfg: SpacecraftActionGroupCfg

    @classmethod
    def mobile_robot_registry(cls) -> Sequence[Type[Spacecraft]]:
        return MobileRobotRegistry.registry.get(MobileRobotType.SPACECRAFT, [])  # type: ignore
