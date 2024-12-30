from simforge_foundry import models as sf_models

import space_robotics_bench.core.sim as sim_utils
from space_robotics_bench.core.asset import AssetBaseCfg, Surface
from space_robotics_bench.core.sim import SimforgeAssetCfg


class MoonSurface(Surface):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/terrain",
        spawn=SimforgeAssetCfg(
            assets=[sf_models.MoonSurface()],
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


class MarsSurface(Surface):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/terrain",
        spawn=SimforgeAssetCfg(
            assets=[sf_models.MarsSurface()],
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
