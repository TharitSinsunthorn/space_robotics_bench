import torch

from srb.core.asset import AssetVariant, Object, Robot, Terrain
from srb.core.env.common.domain import Domain
from srb.core.visuals import VisualsCfg
from srb.utils.cfg import configclass

from .event_cfg import BaseEventCfg


@configclass
class BaseEnvCfg:
    ## Scenario
    seed: int = 0
    domain: Domain = Domain.MOON

    ## Assets
    robot: Robot | AssetVariant | None = AssetVariant.DATASET
    obj: Object | AssetVariant | None = AssetVariant.PROCEDURAL
    terrain: Terrain | AssetVariant | None = AssetVariant.PROCEDURAL

    ## Visuals
    visuals: VisualsCfg = VisualsCfg()

    ## Events
    events: BaseEventCfg = BaseEventCfg()

    ## Misc
    enable_truncation: bool = True

    def __post_init__(self):
        ## Events
        # Gravity
        if self.domain.gravity_variation == 0.0:
            self.events.randomize_gravity = None
        elif self.events.randomize_gravity:
            gravity_z_range = self.domain.gravity_range
            self.events.randomize_gravity.params["distribution_params"] = (
                (0, 0, -gravity_z_range[0]),
                (0, 0, -gravity_z_range[1]),
            )
        # Light
        ## Events
        if self.domain == Domain.ORBIT and self.events.randomize_sunlight_orientation:
            self.events.randomize_sunlight_orientation.params[
                "orientation_distribution_params"
            ] = {
                "roll": (-20.0 * torch.pi / 180.0, -20.0 * torch.pi / 180.0),
                "pitch": (50.0 * torch.pi / 180.0, 50.0 * torch.pi / 180.0),
            }
        if self.domain.light_intensity_variation == 0.0:
            self.events.randomize_sunlight_intensity = None
        elif self.events.randomize_sunlight_intensity:
            self.events.randomize_sunlight_intensity.params["distribution_params"] = (
                self.domain.light_intensity_range
            )
        if self.domain.light_angular_diameter_range == 0.0:
            self.events.randomize_sunlight_angular_diameter = None
        elif self.events.randomize_sunlight_angular_diameter:
            self.events.randomize_sunlight_angular_diameter.params[
                "distribution_params"
            ] = self.domain.light_angular_diameter_range
        if self.domain.light_color_temperature_variation == 0.0:
            self.events.randomize_sunlight_color_temperature = None
        elif self.events.randomize_sunlight_color_temperature:
            self.events.randomize_sunlight_color_temperature.params[
                "distribution_params"
            ] = self.domain.light_color_temperature_range
