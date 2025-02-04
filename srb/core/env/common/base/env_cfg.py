import math
from dataclasses import MISSING

import torch

from srb import assets
from srb.core.asset import AssetVariant, Object, Robot, Terrain
from srb.core.domain import Domain
from srb.core.sim import PhysxCfg, RenderCfg, RigidBodyMaterialCfg, SimulationCfg
from srb.core.visuals import VisualsCfg
from srb.utils.cfg import configclass

from .event_cfg import BaseEventCfg
from .scene_cfg import BaseSceneCfg


@configclass
class BaseEnvCfg:
    ## Scenario
    seed: int = 0
    domain: Domain = Domain.MOON

    ## Assets
    robot: Robot | AssetVariant = AssetVariant.DATASET
    terrain: Terrain | AssetVariant | None = AssetVariant.PROCEDURAL
    obj: Object | AssetVariant | None = AssetVariant.PROCEDURAL

    ## Scene
    scene: BaseSceneCfg = BaseSceneCfg()
    stack: bool = True
    _original_env_spacing: float | None = None

    ## Events
    events: BaseEventCfg = BaseEventCfg()

    ## Time
    env_rate: float = MISSING  # type: ignore
    agent_rate: float = MISSING  # type: ignore

    ## Simulation
    sim = SimulationCfg(
        dt=MISSING,  # type: ignore
        render_interval=MISSING,  # type: ignore
        gravity=MISSING,  # type: ignore
        disable_contact_processing=True,
        physx=PhysxCfg(
            min_position_iteration_count=2,
            min_velocity_iteration_count=1,
            enable_ccd=True,
            enable_stabilization=True,
            bounce_threshold_velocity=0.0,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.005,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        render=RenderCfg(
            enable_translucency=True,
            enable_reflections=True,
        ),
    )

    ## Visuals
    visuals: VisualsCfg = VisualsCfg()

    ## Misc
    enable_truncation: bool = True

    def __post_init__(self):
        ## Scene
        if not self._original_env_spacing:
            self._original_env_spacing = self.scene.env_spacing
        self.scene.env_spacing = 0.0 if self.stack else self._original_env_spacing

        ## Assets -> Scene
        self.scene.light = assets.sunlight_from_cfg(self)
        self.scene.sky = assets.sky_from_cfg(self)
        if isinstance(self.robot, Robot):
            self.scene.robot = self.robot.asset_cfg
            self.actions = self.robot.action_cfg
        if isinstance(self.terrain, Terrain):
            self.scene.terrain = self.terrain.asset_cfg
        if isinstance(self.obj, Object):
            self.scene.obj = self.obj.asset_cfg

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

        ## Simulation
        self.decimation = math.floor(self.agent_rate / self.env_rate)
        self.sim.dt = self.env_rate
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -self.domain.gravity_magnitude)
        _mem_fac = math.floor(self.scene.num_envs**0.25)
        self.sim.physx.gpu_max_rigid_contact_count = 2 ** (12 + _mem_fac)
        self.sim.physx.gpu_max_rigid_patch_count = 2 ** (11 + _mem_fac)
        self.sim.physx.gpu_found_lost_pairs_capacity = 2 ** (16 + _mem_fac)
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** (20 + _mem_fac)
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2 ** (16 + _mem_fac)
        self.sim.physx.gpu_collision_stack_size = 2 ** (17 + _mem_fac)
        self.sim.physx.gpu_heap_capacity = 2 ** (15 + _mem_fac)
        self.sim.physx.gpu_temp_buffer_capacity = 2 ** (12 + _mem_fac)
        self.sim.physx.gpu_max_soft_body_contacts = 2 ** (10 + _mem_fac)
        self.sim.physx.gpu_max_particle_contacts = 2 ** (10 + _mem_fac)
        self.sim.physx.gpu_max_num_partitions = min(
            32, 2 ** math.floor(self.scene.num_envs**0.2)
        )
