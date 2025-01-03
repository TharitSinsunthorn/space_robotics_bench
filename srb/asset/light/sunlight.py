from typing import Any, Dict

import srb.core.envs as env_utils
import srb.core.sim as sim_utils
from srb.core.asset import AssetBaseCfg


def sunlight_from_env_cfg(
    env_cfg: env_utils.EnvironmentConfig,
    *,
    prim_path: str = "/World/light",
    spawn_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.DistantLightCfg(
            intensity=env_cfg.domain.light_intensity,
            angle=env_cfg.domain.light_angular_diameter,
            color_temperature=env_cfg.domain.light_color_temperature,
            enable_color_temperature=True,
            **spawn_kwargs,
        ),
        **kwargs,
    )
