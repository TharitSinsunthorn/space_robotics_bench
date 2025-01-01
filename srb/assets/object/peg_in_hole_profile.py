from os import path

from omni.isaac.lab.utils import configclass

import srb.core.sim as sim_utils
from srb.paths import SRB_ASSETS_DIR_SRB_OBJECT


@configclass
class PegProfileCfg(sim_utils.UsdFileCfg):
    usd_path = path.join(
        SRB_ASSETS_DIR_SRB_OBJECT, "peg_in_hole_profile", "profile.usdc"
    )


@configclass
class PegProfileShortCfg(sim_utils.UsdFileCfg):
    usd_path = path.join(
        SRB_ASSETS_DIR_SRB_OBJECT, "peg_in_hole_profile", "profile_short.usdc"
    )


@configclass
class HoleProfileCfg(sim_utils.UsdFileCfg):
    usd_path = path.join(SRB_ASSETS_DIR_SRB_OBJECT, "peg_in_hole_profile", "hole.usdc")
