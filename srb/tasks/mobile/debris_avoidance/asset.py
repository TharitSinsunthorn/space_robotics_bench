from typing import Tuple

from simforge import TexResConfig

from srb import assets
from srb.core.asset import RigidObjectCfg


def debris_cfg(
    *,
    seed: int,
    num_assets: int,
    prim_path: str = "{ENV_REGEX_NS}/object",
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
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

    return asset_cfg
