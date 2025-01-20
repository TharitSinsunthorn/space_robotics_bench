import math
from typing import Tuple

from simforge import TexResConfig

from srb.core.asset import AssetBaseCfg
from srb.core.envs import env_cfg

from .ground_plane import GroundPlane
from .planetary_surface import MarsSurface, MoonSurface


def terrain_from_cfg(
    cfg: env_cfg.EnvironmentConfig,
    *,
    seed: int,
    num_assets: int,
    prim_path: str = "{ENV_REGEX_NS}/terrain",
    scale: Tuple[float, float] | Tuple[float, float, float] = (10.0, 10.0),
    texture_resolution: TexResConfig | None = None,
    density: float | None = None,
    flat_area_size: int | None = None,
    **kwargs,
) -> AssetBaseCfg | None:
    if not cfg.domain.requires_terrain:
        return None

    match cfg.assets.terrain.variant:
        case env_cfg.AssetVariant.NONE:
            return None

        case env_cfg.AssetVariant.PRIMITIVE:
            asset_cfg = GroundPlane(**kwargs).asset_cfg

            scaling_factor = math.sqrt(num_assets)
            asset_cfg.spawn.size = (  # type: ignore
                scaling_factor * scale[0],
                scaling_factor * scale[1],
            )

        case env_cfg.AssetVariant.PROCEDURAL:
            if len(scale) == 2:
                scale = (scale[0], scale[1], (scale[0] + scale[1]) / 20.0)

            match cfg.domain:
                case env_cfg.Domain.MOON:
                    asset_cfg = MoonSurface(
                        scale=scale,
                        texture_resolution=texture_resolution,
                        density=density,
                        flat_area_size=flat_area_size,
                        **kwargs,
                    ).asset_cfg

                case env_cfg.Domain.MARS:
                    asset_cfg = MarsSurface(
                        scale=scale,
                        texture_resolution=texture_resolution,
                        density=density,
                        flat_area_size=flat_area_size,
                        **kwargs,
                    ).asset_cfg

            asset_cfg.prim_path = prim_path

            asset_cfg.spawn.num_assets = num_assets  # type: ignore
            asset_cfg.spawn.seed = seed  # type: ignore

    return asset_cfg
