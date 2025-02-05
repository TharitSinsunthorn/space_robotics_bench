from dataclasses import MISSING

import torch
from simforge import BakeType

from srb import assets
from srb.core.asset import (
    AssetBaseCfg,
    AssetVariant,
    SingleArmManipulator,
    StaticVehicle,
)
from srb.core.env import BaseEventCfg, BaseSceneCfg, DirectEnvCfg, ViewerCfg
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import FRAME_MARKER_SMALL_CFG
from srb.core.mdp import reset_joints_by_offset
from srb.core.sensor import ContactSensorCfg, FrameTransformerCfg, OffsetCfg
from srb.utils.cfg import configclass


@configclass
class SingleArmSceneCfg(BaseSceneCfg):
    env_spacing = 8.0

    ## Assets
    vehicle: AssetBaseCfg | None = None

    ## Sensors
    tf_robot_ee: FrameTransformerCfg = FrameTransformerCfg(
        prim_path=MISSING,  # type: ignore
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="robot_ee",
                prim_path=MISSING,  # type: ignore
            ),
        ],
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/robot_ee"),
    )
    contacts_robot_arm: ContactSensorCfg = ContactSensorCfg(
        prim_path=MISSING,  # type: ignore
    )


@configclass
class SingleArmEventCfg(BaseEventCfg):
    randomize_robot_joints: EventTermCfg = EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-torch.pi / 32, torch.pi / 32),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class SingleArmEnvCfg(DirectEnvCfg):
    ## Assets
    robot: SingleArmManipulator | AssetVariant = assets.Franka()
    vehicle: StaticVehicle | AssetVariant | None = assets.ConstructionRover()

    ## Scene
    scene: SingleArmSceneCfg = SingleArmSceneCfg()

    ## Events
    events: SingleArmEventCfg = SingleArmEventCfg()

    ## Time
    env_rate: float = 1.0 / 200.0
    agent_rate: float = 1.0 / 50.0

    ## Viewer
    viewer = ViewerCfg(
        eye=(2.0, 0.0, 1.75),
        lookat=(0.0, 0.0, 0.25),
        origin_type="env",
    )

    def __post_init__(self):
        super().__post_init__()

        ## Assets -> Scene
        # Robot
        if isinstance(self.robot, AssetVariant):
            # TODO: Implement SingleArmManipulator from AssetVariant
            raise NotImplementedError()
            self.robot: SingleArmManipulator = ...
            self.scene.robot = self.robot.asset_cfg
            self.actions = self.robot.action_cfg
        # Terrain
        self.scene.terrain = assets.terrain_from_cfg(
            self,
            seed=self.seed,
            num_assets=1 if self.stack else self.scene.num_envs,
            prim_path="/World/terrain" if self.stack else "{ENV_REGEX_NS}/terrain",
            scale=(8.0, 8.0),
            density=0.1,
            texture_resolution={
                BakeType.ALBEDO: 2048,
                BakeType.NORMAL: 1024,
                BakeType.ROUGHNESS: 512,
            },
            flat_area_size=2.0,
        )
        # Vehicle
        if isinstance(self.vehicle, AssetVariant):
            self.vehicle = assets.vehicle_from_cfg(self)
        if isinstance(self.vehicle, StaticVehicle):
            self.vehicle.asset_cfg.prim_path = (
                "/World/vehicle" if self.stack else "{ENV_REGEX_NS}/vehicle"
            )
            self.scene.vehicle = self.vehicle.asset_cfg
            self.scene.vehicle.init_state.pos = (
                self.vehicle.frame_manipulator_base.offset.translation
            )
            self.scene.robot.init_state.pos = (
                self.vehicle.frame_manipulator_base.offset.translation
            )
            self.scene.robot.init_state.rot = (
                self.vehicle.frame_manipulator_base.offset.rotation
            )
        # Sensor - TF (robot EE)
        self.scene.tf_robot_ee.prim_path = (
            f"{self.scene.robot.prim_path}/{self.robot.frame_base.prim_relpath}"
        )
        self.scene.tf_robot_ee.target_frames[
            0
        ].prim_path = f"{self.scene.robot.prim_path}/{self.robot.frame_ee.prim_relpath}"
        self.scene.tf_robot_ee.target_frames[0].offset = OffsetCfg(
            pos=self.robot.frame_ee.offset.translation,
            rot=self.robot.frame_ee.offset.rotation,
        )
        # Sensor - Contact (robot)
        self.scene.contacts_robot_arm.prim_path = (
            f"{self.scene.robot.prim_path}/{self.robot.regex_links_arm}"
        )

        ## Events
        self.events.randomize_robot_joints.params[
            "asset_cfg"
        ].joint_names = self.robot.regex_joints_arm
