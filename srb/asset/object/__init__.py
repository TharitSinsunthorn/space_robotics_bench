from typing import Tuple

from simforge import TexResConfig

from srb.core.asset import RigidObjectCfg
from srb.core.envs import env_cfg

from .asteroid import Asteroid  # noqa: F401
from .cubesat_debris import CubesatDebris  # noqa: F401
from .peg_in_hole_procgen import RandomHole, RandomPeg  # noqa: F401
from .peg_in_hole_profile import ProfileHole, ProfilePeg, ShortProfilePeg  # noqa: F401
from .rock_procgen import LunarRock, MarsRock  # noqa: F401
from .sample_tube import SampleTube  # noqa: F401
from .shape import RandomShape  # noqa: F401
from .solar_panel import SolarPanel  # noqa: F401
from .static_vehicle import *  # noqa: F403


def rigid_object_from_cfg(
    cfg: env_cfg.EnvironmentConfig,
    *,
    seed: int,
    num_assets: int,
    prim_path: str = "{ENV_REGEX_NS}/object",
    scale: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    texture_resolution: TexResConfig | None = None,
    **kwargs,
) -> RigidObjectCfg:
    match cfg.assets.object.variant:
        case env_cfg.AssetVariant.PRIMITIVE:
            asset_cfg = RandomShape(scale=scale).asset_cfg

        case env_cfg.AssetVariant.DATASET:
            match cfg.domain:
                case env_cfg.Domain.MARS:
                    asset_cfg = SampleTube().asset_cfg
                case _:
                    asset_cfg = ShortProfilePeg().asset_cfg

        case env_cfg.AssetVariant.PROCEDURAL:
            match cfg.domain:
                case env_cfg.Domain.ORBIT:
                    asset_cfg = CubesatDebris(
                        scale=scale, texture_resolution=texture_resolution
                    ).asset_cfg

                case env_cfg.Domain.MOON:
                    asset_cfg = LunarRock(
                        scale=scale, texture_resolution=texture_resolution
                    ).asset_cfg

                case env_cfg.Domain.MARS:
                    asset_cfg = MarsRock(
                        scale=scale, texture_resolution=texture_resolution
                    ).asset_cfg

            asset_cfg.spawn.seed = seed  # type: ignore
            asset_cfg.spawn.num_assets = num_assets  # type: ignore

    asset_cfg.prim_path = prim_path
    asset_cfg.spawn.replace(**kwargs)

    return asset_cfg
