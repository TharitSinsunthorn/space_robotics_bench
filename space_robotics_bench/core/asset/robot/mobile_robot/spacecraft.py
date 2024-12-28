from __future__ import annotations

from typing import Sequence, Type

from space_robotics_bench.core.actions import SpacecraftActionGroupCfg
from space_robotics_bench.core.asset.robot.mobile_robot.mobile_robot import MobileRobot
from space_robotics_bench.core.asset.robot.mobile_robot.mobile_robot_type import (
    MobileRobotType,
)


class Spacecraft(
    MobileRobot,
    mobile_robot_entrypoint=MobileRobotType.SPACECRAFT,
    arbitrary_types_allowed=True,
):
    action_cfg: SpacecraftActionGroupCfg

    @classmethod
    def mobile_robot_registry(cls) -> Sequence[Type[Spacecraft]]:
        return super().mobile_robot_registry().get(MobileRobotType.SPACECRAFT, [])  # type: ignore
