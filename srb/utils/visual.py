from typing import Tuple

from srb.core.envs import BaseEnvCfg, Domain
from srb.core.sim import PreviewSurfaceCfg


def contrastive_color_from_env_cfg(
    env_cfg: BaseEnvCfg,
) -> Tuple[float, float, float]:
    match env_cfg.domain:
        case Domain.ASTEROID | Domain.MOON:
            return (0.8, 0.8, 0.8)
        case Domain.EARTH | Domain.MARS | Domain.ORBIT:
            return (0.1, 0.1, 0.1)
        case _:
            return (0.7071, 0.7071, 0.7071)


def preview_surface_from_env_cfg(
    env_cfg: BaseEnvCfg,
) -> PreviewSurfaceCfg:
    return PreviewSurfaceCfg(
        diffuse_color=contrastive_color_from_env_cfg(env_cfg),
    )
