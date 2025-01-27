from typing import TYPE_CHECKING

from srb.core.asset import AssetBaseCfg
from srb.core.sim import DistantLightCfg

if TYPE_CHECKING:
    from srb._typing import AnyEnvCfg


def sunlight_from_cfg(
    cfg: "AnyEnvCfg",
    *,
    prim_path: str = "/World/light",
    **kwargs,
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=DistantLightCfg(
            intensity=cfg.domain.light_intensity,
            angle=cfg.domain.light_angular_diameter,
            color_temperature=cfg.domain.light_color_temperature,
            enable_color_temperature=True,
            **kwargs,
        ),
    )
