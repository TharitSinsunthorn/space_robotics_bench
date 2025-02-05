from dataclasses import MISSING

from simforge import BakeType

from srb import assets
from srb.core.asset import AssetVariant, LeggedRobot
from srb.core.env import BaseEventCfg, BaseSceneCfg, DirectEnvCfg, ViewerCfg
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import (
    push_by_setting_velocity,
    reset_joints_by_scale,
    reset_root_state_uniform,
)
from srb.core.sensor import ContactSensorCfg
from srb.utils.cfg import configclass


@configclass
class LocomotionSceneCfg(BaseSceneCfg):
    env_spacing = 64.0

    contacts_robot: ContactSensorCfg = ContactSensorCfg(
        prim_path=MISSING,  # type: ignore
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )


@configclass
class LocomotionEventCfg(BaseEventCfg):
    randomize_robot_state: EventTermCfg = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    randomize_robot_joints: EventTermCfg = EventTermCfg(
        func=reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot: EventTermCfg = EventTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class LocomotionEnvCfg(DirectEnvCfg):
    ## Assets
    robot: LeggedRobot | AssetVariant = assets.AnymalMulti()

    ## Scene
    scene: LocomotionSceneCfg = LocomotionSceneCfg()

    ## Events
    events: LocomotionEventCfg = LocomotionEventCfg()

    ## Time
    env_rate: float = 1.0 / 200.0
    agent_rate: float = 1.0 / 100.0

    ## Viewer
    viewer = ViewerCfg(
        eye=(4.0, -4.0, 4.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="env",
    )

    def __post_init__(self):
        super().__post_init__()

        ## Assets -> Scene
        # Robot
        if isinstance(self.robot, AssetVariant):
            # TODO: Implement LeggedRobot from AssetVariant
            raise NotImplementedError()
            self.robot: LeggedRobot = ...
            self.scene.robot = self.robot.asset_cfg
            self.actions = self.robot.action_cfg
        # Terrain
        self.scene.terrain = assets.terrain_from_cfg(
            self,
            seed=self.seed,
            num_assets=1 if self.stack else self.scene.num_envs,
            prim_path="/World/terrain" if self.stack else "{ENV_REGEX_NS}/terrain",
            scale=(
                64.0,
                64.0,
            ),
            density=0.2,
            texture_resolution={
                BakeType.ALBEDO: 4096,
                BakeType.NORMAL: 2048,
                BakeType.ROUGHNESS: 1024,
            },
            flat_area_size=4.0,
        )
        # Sensor - Contact (robot)
        # TODO: Check only for feel contacts???
        self.scene.contacts_robot.prim_path = f"{self.scene.robot.prim_path}/.*"
