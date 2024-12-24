from dataclasses import MISSING

from omni.isaac.lab.utils import configclass

from space_robotics_bench.core.actions import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)


@configclass
class ManipulatorTaskSpaceActionCfg:
    arm: DifferentialInverseKinematicsActionCfg = MISSING
    hand: BinaryJointPositionActionCfg = MISSING
