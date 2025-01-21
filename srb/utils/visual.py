from typing import Tuple

import srb.core.envs as env_utils
from srb.core import sim as sim_utils


def contrastive_color_from_env_cfg(
    env_cfg: env_utils.EnvironmentConfig,
) -> Tuple[float, float, float]:
    match env_cfg.domain:
        case env_utils.Domain.ASTEROID | env_utils.Domain.MOON:
            return (0.8, 0.8, 0.8)
        case env_utils.Domain.EARTH | env_utils.Domain.MARS | env_utils.Domain.ORBIT:
            return (0.1, 0.1, 0.1)
        case _:
            return (0.7071, 0.7071, 0.7071)


def preview_surface_from_env_cfg(
    env_cfg: env_utils.EnvironmentConfig,
) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=contrastive_color_from_env_cfg(env_cfg),
    )
