from typing import Any, Dict

import space_robotics_bench.core.asset as asset_utils
import space_robotics_bench.core.envs as env_utils

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
            match env_cfg.scenario:
                case env_utils.Scenario.MOON | env_utils.Scenario.MARS:
                    return ConstructionRover()
                case env_utils.Scenario.ORBIT:
                    return Gateway()
