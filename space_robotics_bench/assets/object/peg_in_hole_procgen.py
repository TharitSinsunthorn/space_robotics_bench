from typing import Sequence

from omni.isaac.lab.utils import configclass
from simforge import Asset
from simforge_foundry import geometry as sf_geometry

from space_robotics_bench.core.sim import SimforgeAssetCfg


@configclass
class PegProcgenCfg(SimforgeAssetCfg):
    assets: Sequence[Asset] = [sf_geometry.PegGeo()]


@configclass
class HoleProcgenCfg(SimforgeAssetCfg):
    assets: Sequence[Asset] = [sf_geometry.HoleGeo()]
