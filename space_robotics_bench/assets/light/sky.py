from os import path
from typing import Any, Dict

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

import space_robotics_bench.core.envs as env_utils
import space_robotics_bench.core.sim as sim_utils
from space_robotics_bench.core.asset import AssetBaseCfg
from space_robotics_bench.paths import SRB_ASSETS_DIR_SRB_HDRI
from space_robotics_bench.utils import rtx_settings


def sky_from_env_cfg(
    env_cfg: env_utils.EnvironmentConfig,
    *,
    prim_path: str = "/World/sky",
    spawn_kwargs: Dict[str, Any] = {},
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

    match env_cfg.scenario:
        case env_utils.Scenario.EARTH:
            texture_file = f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
        case env_utils.Scenario.MARS:
            rtx_settings.simple_fog(
                color=(0.8, 0.4, 0.2),
                intensity=0.25,
                start_height=16.0,
                height_density=0.5,
                fog_distance_density=0.05,
            )
            texture_file = path.join(SRB_ASSETS_DIR_SRB_HDRI, "martian_sky_day.hdr")
        case env_utils.Scenario.ORBIT:
            texture_file = path.join(SRB_ASSETS_DIR_SRB_HDRI, "low_lunar_orbit.jpg")

    if texture_file is None:
        return None
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.DomeLightCfg(
            intensity=0.25 * env_cfg.scenario.light_intensity,
            texture_file=texture_file,
            **spawn_kwargs,
        ),
        **kwargs,
    )
