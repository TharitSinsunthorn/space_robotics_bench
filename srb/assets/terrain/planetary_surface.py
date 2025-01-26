import simforge_foundry

from srb.core.asset import AssetBaseCfg, Surface
from srb.core.sim import CollisionPropertiesCfg, SimforgeAssetCfg


class MoonSurface(Surface):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/terrain",
        spawn=SimforgeAssetCfg(
            assets=[simforge_foundry.MoonSurface()],
            collision_props=CollisionPropertiesCfg(),
        ),
    )


class MarsSurface(Surface):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/terrain",
        spawn=SimforgeAssetCfg(
            assets=[simforge_foundry.MarsSurface()],
            collision_props=CollisionPropertiesCfg(),
        ),
    )
