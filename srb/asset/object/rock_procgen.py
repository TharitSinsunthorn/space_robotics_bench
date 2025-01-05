from typing import Sequence

from omni.isaac.lab.utils import configclass
from simforge import Asset
from simforge_foundry import model as sf_model

from srb.core.sim import SimforgeAssetCfg


@configclass
class LunarRockCfg(SimforgeAssetCfg):
    assets: Sequence[Asset] = [sf_model.MoonRock()]


@configclass
class MarsRockCfg(SimforgeAssetCfg):
    assets: Sequence[Asset] = [sf_model.MarsRock()]
