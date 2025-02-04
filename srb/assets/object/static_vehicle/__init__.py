from typing import TYPE_CHECKING

from srb.core.asset import StaticVehicle
from srb.core.domain import Domain

from .construction_rover import ConstructionRover
from .gateway import Gateway

if TYPE_CHECKING:
    from srb.core.env import SingleArmEnvCfg


def vehicle_from_cfg(
    cfg: "SingleArmEnvCfg",
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
