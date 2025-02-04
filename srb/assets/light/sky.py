from typing import TYPE_CHECKING

from srb.core.asset import AssetBaseCfg
from srb.core.env.common.domain import Domain
from srb.core.sim import DomeLightCfg
from srb.core.visuals import rtx_post
from srb.utils.nucleus import ISAAC_NUCLEUS_DIR
from srb.utils.path import SRB_ASSETS_DIR_SRB_HDRI

if TYPE_CHECKING:
    from srb._typing import AnyEnvCfg


def sky_from_cfg(
    cfg: "AnyEnvCfg",
    *,
    prim_path: str = "/World/sky",
    **kwargs,
) -> AssetBaseCfg | None:
    texture_file = None

    match cfg.domain:
        case Domain.EARTH:
            texture_file = f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
        case Domain.MARS:
            texture_file = SRB_ASSETS_DIR_SRB_HDRI.joinpath(
                "martian_sky_day.hdr"
            ).as_posix()
            rtx_post.fog(
                enable=True,
                color=(0.8, 0.4, 0.2),
                intensity=0.25,
                start_height=16.0,
                height_density=0.5,
                fog_distance_density=0.05,
            )
        case Domain.ORBIT:
            texture_file = SRB_ASSETS_DIR_SRB_HDRI.joinpath(
                "low_lunar_orbit.jpg"
            ).as_posix()

    if texture_file is None:
        return None
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=DomeLightCfg(
            intensity=0.25 * cfg.domain.light_intensity,
            texture_file=texture_file,
            **kwargs,
        ),
    )
