from typing import Sequence

from omni.isaac.lab.utils import configclass
from simforge.core import Asset
from simforge.integrations.isaaclab import SimforgeAssetCfg
from simforge_foundry import geometry as sf_geometry


@configclass
class PegProcgenCfg(SimforgeAssetCfg):
    assets: Sequence[Asset] = [sf_geometry.PegGeo()]


@configclass
class HoleProcgenCfg(SimforgeAssetCfg):
    assets: Sequence[Asset] = [sf_geometry.HoleGeo()]
