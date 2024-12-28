from pydantic import PositiveFloat
from simforge.integrations.isaaclab import SimforgeAssetCfg
from simforge.typing import TexResConfig
from simforge_foundry import models as sf_models

import space_robotics_bench.core.sim as sim_utils
from space_robotics_bench.core.asset import AssetBaseCfg, Surface


class MoonSurface(Surface):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/terrain",
        spawn=SimforgeAssetCfg(
            assets=[sf_models.MoonSurface()],
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    density: PositiveFloat | None = None
    texture_resolution: TexResConfig | None = None


class MarsSurface(Surface):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/terrain",
        spawn=SimforgeAssetCfg(
            assets=[sf_models.MarsSurface()],
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    density: PositiveFloat | None = None
    texture_resolution: TexResConfig | None = None
