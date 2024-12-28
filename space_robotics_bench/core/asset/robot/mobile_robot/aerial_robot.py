from __future__ import annotations

from typing import Sequence, Type

from space_robotics_bench.core.actions import MultiCopterActionGroupCfg
from space_robotics_bench.core.asset import Frame
from space_robotics_bench.core.asset.robot.mobile_robot.mobile_robot import MobileRobot
from space_robotics_bench.core.asset.robot.mobile_robot.mobile_robot_type import (
    MobileRobotType,
)


class AerialRobot(MobileRobot, mobile_robot_entrypoint=MobileRobotType.AERIAL):
    @classmethod
    def mobile_robot_registry(cls) -> Sequence[Type[AerialRobot]]:
        return super().mobile_robot_registry().get(MobileRobotType.AERIAL, [])  # type: ignore


class MultiCopter(AerialRobot, arbitrary_types_allowed=True):
    ## Actions
    action_cfg: MultiCopterActionGroupCfg

    ## Frames
    frame_camera_bottom: Frame

    ## Links
    regex_links_rotors: str

    ## Joints
    regex_joints_rotors: str
