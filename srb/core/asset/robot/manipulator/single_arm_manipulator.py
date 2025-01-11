from __future__ import annotations

from typing import Sequence, Type

from srb.core.actions import ManipulatorTaskSpaceActionCfg
from srb.core.asset.common import Frame
from srb.core.asset.robot.manipulator.manipulator import (
    Manipulator,
    ManipulatorRegistry,
)
from srb.core.asset.robot.manipulator.manipulator_type import ManipulatorType


class SingleArmManipulator(
    Manipulator, manipulator_entrypoint=ManipulatorType.SINGLE_ARM
):
    ## Actions
    action_cfg: ManipulatorTaskSpaceActionCfg

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
