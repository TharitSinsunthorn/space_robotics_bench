from srb._typing import AnyEnvCfg
from srb.core.asset import AssetBaseCfg
from srb.core.env import Domain
from srb.core.sim import DomeLightCfg
from srb.utils import rtx_settings
from srb.utils.nucleus import ISAAC_NUCLEUS_DIR
from srb.utils.path import SRB_ASSETS_DIR_SRB_HDRI


def sky_from_cfg(
    cfg: AnyEnvCfg,
    *,
    prim_path: str = "/World/sky",
    **kwargs,
) -> AssetBaseCfg | None:
    texture_file = None

    # Enable rendering effects
    rtx_settings.auto_exposure()
    # TODO: Update/find a better place for rendering effects
    # rtx_settings.chromatic_aberration(strength=(0.2 * -0.055, 0.2 * -0.075, 0.2 * 0.015))
    # rtx_settings.depth_of_field_override(subject_distance=10.0)
    # rtx_settings.lens_flare_physical(
    #     scale=1.0,
    #     blades=0,
    #     cutoff_fuzziness=0.2,
    #     noise_strength=0.025,
    #     dust_strength=0.1,
    #     scratch_strength=0.05,
    # )
    # rtx_settings.tv_noise(
    #     enable_scanlines=True,
    #     enable_scroll_bug=True,
    #     enable_vignetting=True,
    #     enable_vignetting_flickering=True,
    #     enable_ghost_flickering=True,
    #     enable_wave_distortion=True,
    #     enable_vertical_lines=True,
    #     enable_random_splotches=True,
    #     enable_film_grain=True,
    # )

    match cfg.domain:
        case Domain.EARTH:
            texture_file = f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
        case Domain.MARS:
            # Move elsewhere
            rtx_settings.simple_fog(
                color=(0.8, 0.4, 0.2),
                intensity=0.25,
                start_height=16.0,
                height_density=0.5,
                fog_distance_density=0.05,
            )
            texture_file = SRB_ASSETS_DIR_SRB_HDRI.joinpath(
                "martian_sky_day.hdr"
            ).as_posix()
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
