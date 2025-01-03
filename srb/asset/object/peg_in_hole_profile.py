from omni.isaac.lab.utils import configclass

import srb.core.sim as sim_utils
from srb.utils.path import SRB_ASSETS_DIR_SRB_OBJECT


@configclass
class PegProfileCfg(sim_utils.UsdFileCfg):
    usd_path = (
        SRB_ASSETS_DIR_SRB_OBJECT.joinpath("peg_in_hole_profile")
        .joinpath("profile.usdc")
        .as_posix()
    )


@configclass
class PegProfileShortCfg(sim_utils.UsdFileCfg):
    usd_path = (
        SRB_ASSETS_DIR_SRB_OBJECT.joinpath("peg_in_hole_profile")
        .joinpath("profile_short.usdc")
        .as_posix()
    )


@configclass
class HoleProfileCfg(sim_utils.UsdFileCfg):
    usd_path = (
        SRB_ASSETS_DIR_SRB_OBJECT.joinpath("peg_in_hole_profile")
        .joinpath("hole.usdc")
        .as_posix()
    )
