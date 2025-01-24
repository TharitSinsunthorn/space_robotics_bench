from srb.core.asset import AssetBaseCfg, Surface
from srb.core.sim import GroundPlaneCfg


class GroundPlane(Surface):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/terrain",
        spawn=GroundPlaneCfg(
            color=(0.0, 158.0 / 255.0, 218.0 / 255.0),
        ),
    )
