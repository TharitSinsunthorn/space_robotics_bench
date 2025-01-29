import math

import torch
from simforge import BakeType

from srb import assets
from srb.core.asset import AssetVariant, Manipulator, StaticVehicle
from srb.core.env import (
    BaseEventCfg,
    DirectEnvCfg,
    Domain,
    InteractiveSceneCfg,
    ViewerCfg,
)
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import FRAME_MARKER_SMALL_CFG
from srb.core.mdp import reset_joints_by_offset
from srb.core.sensor import ContactSensorCfg, FrameTransformerCfg, OffsetCfg
from srb.core.sim import PhysxCfg, RenderCfg, RigidBodyMaterialCfg, SimulationCfg
from srb.utils.cfg import configclass


@configclass
class ManipulationEventCfg(BaseEventCfg):
    ## Robot
    randomize_robot_joints: EventTermCfg | None = EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-torch.pi / 32, torch.pi / 32),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class ManipulationEnvCfg(DirectEnvCfg):
    ## Environment
    episode_length_s: float = 50.0
    env_rate: float = 1.0 / 200.0

    ## Agent
    agent_rate: float = 1.0 / 50.0

    ## Assets
    robot: Manipulator | AssetVariant | None = AssetVariant.DATASET
    vehicle: StaticVehicle | AssetVariant | None = AssetVariant.DATASET

    ## Simulation
    sim = SimulationCfg(
        disable_contact_processing=True,
        physx=PhysxCfg(
            enable_ccd=True,
            enable_stabilization=True,
            bounce_threshold_velocity=0.0,
            friction_correlation_distance=0.005,
            min_velocity_iteration_count=2,
            # GPU settings
            gpu_temp_buffer_capacity=2 ** (24 - 2),
            gpu_max_rigid_contact_count=2 ** (22 - 1),
            gpu_max_rigid_patch_count=2 ** (13 - 0),
            gpu_heap_capacity=2 ** (26 - 3),
            gpu_found_lost_pairs_capacity=2 ** (18 - 1),
            gpu_found_lost_aggregate_pairs_capacity=2 ** (10 - 0),
            gpu_total_aggregate_pairs_capacity=2 ** (10 - 0),
            gpu_max_soft_body_contacts=2 ** (20 - 1),
            gpu_max_particle_contacts=2 ** (20 - 1),
            gpu_collision_stack_size=2 ** (26 - 3),
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
        lookat=(0.0, 0.0, 0.25),
        eye=(2.0, 0.0, 1.75),
        origin_type="env",
        env_index=0,
    )

    ## Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=8.0, replicate_physics=False
    )

    ## Events
    events: ManipulationEventCfg = ManipulationEventCfg()

    def __post_init__(self):
        super().__post_init__()

        ## Simulation
        self.decimation = int(self.agent_rate / self.env_rate)
        self.sim.dt = self.env_rate
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -self.domain.gravity_magnitude)
        # Increase GPU settings based on the number of environments
        gpu_capacity_factor = math.pow(self.scene.num_envs, 0.2)
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
            2 ** math.floor(1.0 + gpu_capacity_factor), 32
        )

        ## Scene
        if self.domain == Domain.ORBIT:
            self.scene.env_spacing = 32.0
        self.scene.light = assets.sunlight_from_cfg(self)
        self.scene.sky = assets.sky_from_cfg(self)
        # self.robot = assets.manipulator_from_env_cfg(self)
        self.robot = assets.Franka()
        self.scene.robot = self.robot.asset_cfg
        self.vehicle = assets.vehicle_from_cfg(self)
        self.scene.terrain = assets.terrain_from_cfg(
            self,
            seed=self.seed,
            num_assets=self.scene.num_envs,
            scale=(
                self.scene.env_spacing - 1.0,
                self.scene.env_spacing - 1.0,
            ),
            density=0.1,
            texture_resolution={
                BakeType.ALBEDO: 2048,
                BakeType.NORMAL: 1024,
                BakeType.ROUGHNESS: 512,
            },
            flat_area_size=2.0,
        )
        if self.vehicle:
            # Add vehicle to scene
            self.scene.vehicle = self.vehicle.asset_cfg
            self.scene.vehicle.init_state.pos = (
                self.vehicle.frame_manipulator_base.offset.translation
            )

            # Update the robot based on the vehicle
            self.scene.robot.init_state.pos = (
                self.vehicle.frame_manipulator_base.offset.translation
            )
            self.scene.robot.init_state.rot = (
                self.vehicle.frame_manipulator_base.offset.rotation
            )

        ## Actions
        self.actions = self.robot.action_cfg

        ## Sensors
        self.scene.tf_robot_ee = FrameTransformerCfg(
            prim_path=f"{self.scene.robot.prim_path}/{self.robot.frame_base.prim_relpath}",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    name="robot_ee",
                    prim_path=f"{self.scene.robot.prim_path}/{self.robot.frame_ee.prim_relpath}",
                    offset=OffsetCfg(
                        pos=self.robot.frame_ee.offset.translation,
                        rot=self.robot.frame_ee.offset.rotation,
                    ),
                ),
            ],
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
                prim_path="/Visuals/robot_ee"
            ),
        )
        self.scene.contacts_robot = ContactSensorCfg(
            prim_path=f"{self.scene.robot.prim_path}/.*",
            update_period=0.0,
        )

        ## Events
        self.events.randomize_robot_joints.params[  # type: ignore
            "asset_cfg"
        ].joint_names = self.robot.regex_joints_arm
