from omni.isaac.lab.utils import configclass

import srb.core.sim as sim_utils
from srb.paths import SRB_ASSETS_DIR_SRB_OBJECT


@configclass
class SolarPanelCfg(sim_utils.UsdFileCfg):
    usd_path = (
        SRB_ASSETS_DIR_SRB_OBJECT.joinpath("solar_panel")
        .joinpath("solar_panel.usdc")
        .as_posix()
    )
