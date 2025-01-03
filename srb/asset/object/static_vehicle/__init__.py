from typing import Any, Dict

import srb.core.asset as asset_utils
import srb.core.envs as env_utils

from .construction_rover import ConstructionRover
from .gateway import Gateway


def vehicle_from_env_cfg(
    env_cfg: env_utils.EnvironmentConfig,
    *,
    prim_path: str = "{ENV_REGEX_NS}/vehicle",
    spawn_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> asset_utils.StaticVehicle | None:
    match env_cfg.assets.vehicle.variant:
        case env_utils.AssetVariant.NONE:
            return None
        case _:
            match env_cfg.domain:
                case env_utils.Domain.MOON | env_utils.Domain.MARS:
                    return ConstructionRover()
                case env_utils.Domain.ORBIT:
                    return Gateway()
