import srb.core.asset as asset_utils
from srb.core.envs import env_cfg

from .construction_rover import ConstructionRover
from .gateway import Gateway


def vehicle_from_cfg(
    cfg: env_cfg.EnvironmentConfig,
    *,
    prim_path: str = "{ENV_REGEX_NS}/vehicle",
    **kwargs,
) -> asset_utils.StaticVehicle | None:
    match cfg.assets.vehicle.variant:
        case env_cfg.AssetVariant.NONE:
            return None

        case _:
            match cfg.domain:
                case env_cfg.Domain.MOON | env_cfg.Domain.MARS:
                    asset = ConstructionRover(**kwargs)

                case env_cfg.Domain.ORBIT:
                    asset = Gateway(**kwargs)

            asset.asset_cfg.prim_path = prim_path

    return asset
