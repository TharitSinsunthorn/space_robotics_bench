from dataclasses import MISSING

import torch

from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import (  # noqa F401
    follow_xform_orientation_linear_trajectory,
    randomize_gravity_uniform,
    randomize_usd_prim_attribute_uniform,
    reset_scene_to_default,
    reset_xform_orientation_uniform,
)
from srb.utils.cfg import configclass


@configclass
class BaseEventCfg:
    # Default reset
    reset_scene: EventTermCfg | None = EventTermCfg(
        func=reset_scene_to_default, mode="reset"
    )

    # Gravity
    randomize_gravity: EventTermCfg | None = EventTermCfg(
        func=randomize_gravity_uniform,
        mode="interval",
        is_global_time=True,
        interval_range_s=(10.0, 60.0),
        params={"distribution_params": MISSING},
    )

    # Light
    randomize_sunlight_orientation: EventTermCfg | None = EventTermCfg(
        func=reset_xform_orientation_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("light"),
            "orientation_distribution_params": {
                "roll": (
                    -75.0 * torch.pi / 180.0,
                    75.0 * torch.pi / 180.0,
                ),
                "pitch": (
                    -75.0 * torch.pi / 180.0,
                    75.0 * torch.pi / 180.0,
                ),
            },
        },
    )
    # progress_sunlight_orientation: EventTermCfg | None = EventTermCfg(
    #     func=follow_xform_orientation_linear_trajectory,
    #     mode="interval",
    #     interval_range_s=(0.5, 0.5),
    #     is_global_time=True,
    #     params={
    #         "asset_cfg": SceneEntityCfg("light"),
    #         "orientation_step_params": {
    #             "roll": 0.1 * torch.pi / 180.0,
    #             "pitch": 0.1 * torch.pi / 180.0,
    #         },
    #     },
    # )
    randomize_sunlight_intensity: EventTermCfg | None = EventTermCfg(
        func=randomize_usd_prim_attribute_uniform,
        mode="interval",
        is_global_time=True,
        interval_range_s=(10.0, 60.0),
        params={
            "asset_cfg": SceneEntityCfg("light"),
            "attr_name": "intensity",
            "distribution_params": MISSING,
        },
    )
    randomize_sunlight_angular_diameter: EventTermCfg | None = EventTermCfg(
        func=randomize_usd_prim_attribute_uniform,
        mode="interval",
        is_global_time=True,
        interval_range_s=(10.0, 60.0),
        params={
            "asset_cfg": SceneEntityCfg("light"),
            "attr_name": "angle",
            "distribution_params": MISSING,
        },
    )
    randomize_sunlight_color_temperature: EventTermCfg | None = EventTermCfg(
        func=randomize_usd_prim_attribute_uniform,
        mode="interval",
        is_global_time=True,
        interval_range_s=(10.0, 60.0),
        params={
            "asset_cfg": SceneEntityCfg("light"),
            "attr_name": "color_temperature",
            "distribution_params": MISSING,
        },
    )
