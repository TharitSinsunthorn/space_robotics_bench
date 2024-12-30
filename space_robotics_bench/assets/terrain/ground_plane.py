import space_robotics_bench.core.sim as sim_utils
from space_robotics_bench.core.asset import AssetBaseCfg, Surface


class GroundPlane(Surface):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/terrain",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.0, 158.0 / 255.0, 218.0 / 255.0),
        ),
    )
