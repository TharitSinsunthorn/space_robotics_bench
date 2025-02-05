from typing import Tuple

from simforge import TexResConfig

from srb import assets
from srb.core.asset import RigidObjectCfg


def debris_cfg(
    *,
    seed: int,
    num_assets: int,
    prim_path: str = "{ENV_REGEX_NS}/debris",
    scale: Tuple[float, float, float] = (5.0, 5.0, 5.0),
    texture_resolution: TexResConfig | None = None,
    **kwargs,
) -> RigidObjectCfg:
    asset_cfg = assets.Asteroid(
        scale=scale, texture_resolution=texture_resolution
    ).asset_cfg

    asset_cfg.spawn.seed = seed  # type: ignore
    asset_cfg.spawn.num_assets = num_assets  # type: ignore
    asset_cfg.prim_path = prim_path
    asset_cfg.spawn.replace(**kwargs)

    asset_cfg.spawn.assets[0].geo.ops[0].scale_std = (
        0.2 * scale[0],
        0.2 * scale[1],
        0.2 * scale[2],
    )

    return asset_cfg
