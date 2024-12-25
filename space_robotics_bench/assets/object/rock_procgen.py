from typing import Sequence

from omni.isaac.lab.utils import configclass
from simforge.core import Asset
from simforge.integrations.isaaclab import SimforgeAssetCfg
from simforge_foundry import models as sf_models


@configclass
class LunarRockCfg(SimforgeAssetCfg):
    assets: Sequence[Asset] = [sf_models.LunarRock()]
