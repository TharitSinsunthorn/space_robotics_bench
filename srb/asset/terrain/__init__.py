import math
from typing import Any, Dict, Tuple

import srb.core.envs as env_utils
from srb.core.asset import Terrain

# TODO: Downgrade assets to be spawners instead of asset_base
from .ground_plane import GroundPlane
from .planetary_surface import MarsSurface, MoonSurface


def terrain_from_env_cfg(
    env_cfg: env_utils.EnvironmentConfig,
    *,
    size: Tuple[float, float] = (10.0, 10.0),
    num_assets: int = 1,
    prim_path: str = "{ENV_REGEX_NS}/terrain",
    **kwargs,
) -> Terrain:
    match env_cfg.domain:
        case env_utils.Domain.ORBIT:
            return None

    match env_cfg.assets.terrain.variant:
        case env_utils.AssetVariant.PRIMITIVE:
            asset = GroundPlane(
                scale=(
                    10 * math.sqrt(num_assets) * size[0],
                    10 * math.sqrt(num_assets) * size[1],
                )
            )
            asset.asset_cfg.prim_path = prim_path
            return asset

        case env_utils.AssetVariant.PROCEDURAL:
            match env_cfg.domain:
                case env_utils.Domain.MOON:
                    asset = MoonSurface(
                        scale=(size[0], size[0], (size[0] + size[1]) / 20.0),
                        **kwargs,
                    )

                case env_utils.Domain.MARS:
                    asset = MarsSurface(
                        scale=(size[0], size[0], (size[0] + size[1]) / 20.0),
                        **kwargs,
                    )

            asset.asset_cfg.prim_path = prim_path
            asset.asset_cfg.spawn.num_assets = num_assets
            asset.asset_cfg.spawn.seed = env_cfg.seed
            return asset

    raise NotImplementedError()
