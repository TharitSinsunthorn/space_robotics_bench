import math

import torch
from simforge import BakeType

from srb import asset
from srb.core.envs import DirectEnvCfg, InteractiveSceneCfg, ViewerCfg
from srb.core.managers import EventTermCfg, SceneEntityCfg
from srb.core.mdp import (
    reset_root_state_uniform,
    reset_scene_to_default,
    reset_xform_orientation_uniform,
)
from srb.core.sim import PhysxCfg, RenderCfg, RigidBodyMaterialCfg, SimulationCfg
from srb.utils import configclass


@configclass
class BaseAerialRoboticsEnvEventCfg:
    ## Default scene reset
    reset_all = EventTermCfg(func=reset_scene_to_default, mode="reset")

    ## Light
    reset_rand_light_rot = EventTermCfg(
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
    # reset_rand_light_rot = EventTermCfg(
    #     func=follow_xform_orientation_linear_trajectory,
    #     mode="interval",
    #     interval_range_s=(0.1, 0.1),
    #     is_global_time=True,
    #     params={
    #         "asset_cfg": SceneEntityCfg("light"),
    #         "orientation_step_params": {
    #             "roll": 0.25 * torch.pi / 180.0,
    #             "pitch": 0.5 * torch.pi / 180.0,
    #         },
    #     },
    # )

    ## Robot
    reset_rand_robot_state = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-5.0, 5.0),
                "y": (-5.0, 5.0),
                "z": (10.0, 10.0),
                "yaw": (
                    -torch.pi,
                    torch.pi,
                ),
            },
            "velocity_range": {},
        },
    )


@configclass
class BaseAerialRoboticsEnvCfg(DirectEnvCfg):
    ## Environment
    episode_length_s: float = 50.0
    env_rate: float = 1.0 / 50.0

    ## Agent
    agent_rate: float = 1.0 / 50.0

    ## Simulation
    sim = SimulationCfg(
        disable_contact_processing=True,
        physx=PhysxCfg(
            enable_ccd=False,
            enable_stabilization=False,
            bounce_threshold_velocity=0.0,
            friction_correlation_distance=0.02,
            min_velocity_iteration_count=1,
            # GPU settings
            gpu_temp_buffer_capacity=2 ** (24 - 5),
            gpu_max_rigid_contact_count=2 ** (22 - 4),
            gpu_max_rigid_patch_count=2 ** (13 - 1),
            gpu_heap_capacity=2 ** (26 - 6),
            gpu_found_lost_pairs_capacity=2 ** (18 - 2),
            gpu_found_lost_aggregate_pairs_capacity=2 ** (10 - 1),
            gpu_total_aggregate_pairs_capacity=2 ** (10 - 1),
            gpu_max_soft_body_contacts=2 ** (20 - 3),
            gpu_max_particle_contacts=2 ** (20 - 3),
            gpu_collision_stack_size=2 ** (26 - 4),
            gpu_max_num_partitions=8,
        ),
        render=RenderCfg(
            enable_reflections=True,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
    )

    ## Viewer
    viewer = ViewerCfg(
        lookat=(0.0, 0.0, 4.0),
        eye=(-12.0, 0.0, 32.0),
        origin_type="env",
        env_index=0,
    )

    ## Scene
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=64.0, replicate_physics=False)

    ## Events
    events = BaseAerialRoboticsEnvEventCfg()

    def __post_init__(self):
        super().__post_init__()

        ## Simulation
        self.decimation = int(self.agent_rate / self.env_rate)
        self.sim.dt = self.env_rate
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -self.domain.gravity_magnitude)
        # Increase GPU settings based on the number of environments
        gpu_capacity_factor = self.scene.num_envs
        self.sim.physx.gpu_heap_capacity *= gpu_capacity_factor
        self.sim.physx.gpu_collision_stack_size *= gpu_capacity_factor
        self.sim.physx.gpu_temp_buffer_capacity *= gpu_capacity_factor
        self.sim.physx.gpu_max_rigid_contact_count *= gpu_capacity_factor
        self.sim.physx.gpu_max_rigid_patch_count *= gpu_capacity_factor
        self.sim.physx.gpu_found_lost_pairs_capacity *= gpu_capacity_factor
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity *= gpu_capacity_factor
        self.sim.physx.gpu_total_aggregate_pairs_capacity *= gpu_capacity_factor
        self.sim.physx.gpu_max_soft_body_contacts *= gpu_capacity_factor
        self.sim.physx.gpu_max_particle_contacts *= gpu_capacity_factor
        self.sim.physx.gpu_max_num_partitions = min(
            2 ** math.floor(1.0 + math.pow(self.scene.num_envs, 0.2)), 32
        )

        ## Scene
        self.scene.light = asset.sunlight_from_cfg(self)
        self.scene.sky = asset.sky_from_cfg(self)
        self.scene.terrain = asset.terrain_from_cfg(
            self,
            seed=self.seed,
            num_assets=self.scene.num_envs,
            scale=(
                self.scene.env_spacing - 1.0,
                self.scene.env_spacing - 1.0,
            ),
            density=0.5,
            texture_resolution={
                BakeType.ALBEDO: 4096,
                BakeType.NORMAL: 2048,
                BakeType.ROUGHNESS: 1024,
            },
        )
        self.robot = asset.Ingenuity()
        self.scene.robot = self.robot.asset_cfg

        ## Actions
        self.actions = self.robot.action_cfg
