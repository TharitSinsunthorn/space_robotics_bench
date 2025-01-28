from __future__ import annotations

from typing import Sequence, Type

from srb.core.action import SingleArmIK_BinaryGripper
from srb.core.asset.common import Frame
from srb.core.asset.robot.manipulation.manipulator import (
    Manipulator,
    ManipulatorRegistry,
)
from srb.core.asset.robot.manipulation.manipulator_type import ManipulatorType


class SingleArmManipulator(
    Manipulator, manipulator_entrypoint=ManipulatorType.SINGLE_ARM
):
    ## Actions
    action_cfg: SingleArmIK_BinaryGripper

    ## Frames
    frame_ee: Frame
    frame_camera_base: Frame
    frame_camera_wrist: Frame

    ## Links
    regex_links_arm: str
    regex_links_hand: str

    ## Joints
    regex_joints_arm: str
    regex_joints_hand: str

    @classmethod
    def manipulator_registry(cls) -> Sequence[Type[SingleArmManipulator]]:
        return ManipulatorRegistry.registry.get(ManipulatorType.SINGLE_ARM, [])  # type: ignore
