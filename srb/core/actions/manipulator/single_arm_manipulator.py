from dataclasses import MISSING

from omni.isaac.lab.utils import configclass

from srb.core.actions import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)


@configclass
class ManipulatorTaskSpaceActionCfg:
    arm: DifferentialInverseKinematicsActionCfg = MISSING
    hand: BinaryJointPositionActionCfg = MISSING
