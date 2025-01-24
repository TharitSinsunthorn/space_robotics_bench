from srb.core.asset import AssetBaseCfg
from srb.core.envs import BaseEnvCfg
from srb.core.sim import DistantLightCfg


def sunlight_from_cfg(
    cfg: BaseEnvCfg,
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
