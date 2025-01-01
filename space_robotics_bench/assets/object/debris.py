from typing import Sequence

from omni.isaac.lab.utils import configclass
from simforge import Asset
from simforge_foundry import model as sf_model

from space_robotics_bench.core.sim import SimforgeAssetCfg


@configclass
class CubesatDebris(SimforgeAssetCfg):
    assets: Sequence[Asset] = [sf_model.Cubesat()]
