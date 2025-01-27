from typing import TYPE_CHECKING, Tuple

from srb.core.env import Domain
from srb.core.sim import PreviewSurfaceCfg

if TYPE_CHECKING:
    from srb._typing import AnyEnvCfg


def contrastive_color_from_env_cfg(
    env_cfg: "AnyEnvCfg",
) -> Tuple[float, float, float]:
    match env_cfg.domain:
        case Domain.ASTEROID | Domain.MOON:
            return (0.8, 0.8, 0.8)
        case Domain.EARTH | Domain.MARS | Domain.ORBIT:
            return (0.1, 0.1, 0.1)
        case _:
            return (0.7071, 0.7071, 0.7071)


def preview_surface_from_env_cfg(
    env_cfg: "AnyEnvCfg",
) -> PreviewSurfaceCfg:
    return PreviewSurfaceCfg(
        diffuse_color=contrastive_color_from_env_cfg(env_cfg),
    )
