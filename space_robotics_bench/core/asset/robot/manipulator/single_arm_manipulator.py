from __future__ import annotations

from typing import Sequence, Type

from space_robotics_bench.core.actions import ManipulatorTaskSpaceActionCfg
from space_robotics_bench.core.asset.robot.manipulator.manipulator import Manipulator
from space_robotics_bench.core.asset.robot.manipulator.manipulator_type import (
    ManipulatorType,
)
from space_robotics_bench.core.asset.utils import Frame


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
        return super().manipulator_registry().get(ManipulatorType.SINGLE_ARM, [])  # type: ignore
