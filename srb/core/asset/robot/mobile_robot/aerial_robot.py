from __future__ import annotations

from typing import Sequence, Type

from srb.core.actions import MultiCopterActionGroupCfg
from srb.core.asset import Frame
from srb.core.asset.robot.mobile_robot.mobile_robot import (
    MobileRobot,
    MobileRobotRegistry,
)
from srb.core.asset.robot.mobile_robot.mobile_robot_type import MobileRobotType


class AerialRobot(MobileRobot, mobile_robot_entrypoint=MobileRobotType.AERIAL):
    @classmethod
    def mobile_robot_registry(cls) -> Sequence[Type[AerialRobot]]:
        return MobileRobotRegistry.registry.get(MobileRobotType.AERIAL, [])  # type: ignore


class MultiCopter(
    AerialRobot, mobile_robot_metaclass=True, arbitrary_types_allowed=True
):
    ## Actions
    action_cfg: MultiCopterActionGroupCfg

    ## Frames
    frame_camera_bottom: Frame

    ## Links
    regex_links_rotors: str

    ## Joints
    regex_joints_rotors: str
