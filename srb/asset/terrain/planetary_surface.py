from simforge_foundry import model as sf_model

from srb.core import sim as sim_utils
from srb.core.asset import AssetBaseCfg, Surface
from srb.core.sim import SimforgeAssetCfg


class MoonSurface(Surface):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/terrain",
        spawn=SimforgeAssetCfg(
            assets=[sf_model.MoonSurface()],
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


class MarsSurface(Surface):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/terrain",
        spawn=SimforgeAssetCfg(
            assets=[sf_model.MarsSurface()],
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
