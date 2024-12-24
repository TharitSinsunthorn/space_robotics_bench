from __future__ import annotations

from typing import Sequence, Type

from space_robotics_bench.core.actions import ManipulatorTaskSpaceActionCfg
from space_robotics_bench.core.asset.robot.manipulator.manipulator import Manipulator
from space_robotics_bench.core.asset.robot.manipulator.manipulator_type import (
    ManipulatorType,
)
from space_robotics_bench.core.asset.tf import FrameCfg


class SingleArmManipulator(
    Manipulator, manipulator_entrypoint=ManipulatorType.SINGLE_ARM
):
    ## Actions
    action_cfg: ManipulatorTaskSpaceActionCfg

    ## Frames
    frame_ee: FrameCfg
    frame_camera_base: FrameCfg
    frame_camera_wrist: FrameCfg

    ## Links
    regex_links_arm: str
    regex_links_hand: str

    ## Joints
    regex_joints_arm: str
    regex_joints_hand: str

    @classmethod
    def manipulator_registry(cls) -> Sequence[Type[SingleArmManipulator]]:
        return super().manipulator_registry().get(ManipulatorType.SINGLE_ARM, [])  # type: ignore
