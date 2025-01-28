from __future__ import annotations

from typing import Sequence, Type

from srb.core.action import MultiCopterBodyVelocity
from srb.core.asset import Frame
from srb.core.asset.robot.mobile.mobile_robot import MobileRobot, MobileRobotRegistry
from srb.core.asset.robot.mobile.mobile_robot_type import MobileRobotType


class AerialRobot(MobileRobot, mobile_robot_entrypoint=MobileRobotType.AERIAL):
    @classmethod
    def mobile_robot_registry(cls) -> Sequence[Type[AerialRobot]]:
        return MobileRobotRegistry.registry.get(MobileRobotType.AERIAL, [])  # type: ignore


class MultiCopter(
    AerialRobot, mobile_robot_metaclass=True, arbitrary_types_allowed=True
):
    ## Actions
    action_cfg: MultiCopterBodyVelocity

    ## Frames
    frame_camera_bottom: Frame

    ## Links
    regex_links_rotors: str

    ## Joints
    regex_joints_rotors: str
