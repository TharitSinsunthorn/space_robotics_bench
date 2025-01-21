from srb.core import sim as sim_utils
from srb.core.asset import AssetBaseCfg, Surface


class GroundPlane(Surface):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/terrain",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.0, 158.0 / 255.0, 218.0 / 255.0),
        ),
    )
