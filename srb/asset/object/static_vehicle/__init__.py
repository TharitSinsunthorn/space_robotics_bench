from srb.core.asset import StaticVehicle
from srb.core.envs import Domain
from srb.env.manipulation import BaseManipulationEnvCfg

from .construction_rover import ConstructionRover
from .gateway import Gateway


def vehicle_from_cfg(
    cfg: BaseManipulationEnvCfg,
    *,
    prim_path: str = "{ENV_REGEX_NS}/vehicle",
    **kwargs,
) -> StaticVehicle | None:
    match cfg.vehicle:
        case None:
            return None

        case _:
            match cfg.domain:
                case Domain.MOON | Domain.MARS:
                    asset = ConstructionRover(**kwargs)

                case Domain.ORBIT:
                    asset = Gateway(**kwargs)

            asset.asset_cfg.prim_path = prim_path

    return asset
