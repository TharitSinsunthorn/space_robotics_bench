import torch
from simforge import BakeType

from srb import assets
from srb.core.asset import AssetVariant, WheeledRobot
from srb.core.env import BaseEventCfg, BaseSceneCfg, DirectEnvCfg, ViewerCfg
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import reset_root_state_uniform
from srb.utils.cfg import configclass


@configclass
class WheeledSceneCfg(BaseSceneCfg):
    env_spacing = 64.0


@configclass
class WheeledEventCfg(BaseEventCfg):
    randomize_robot_state: EventTermCfg = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "yaw": (
                    -torch.pi,
                    torch.pi,
                ),
            },
            "velocity_range": {},
        },
    )


@configclass
class WheeledEnvCfg(DirectEnvCfg):
    ## Assets
    robot: WheeledRobot | AssetVariant = assets.Perseverance()

    ## Scene
    scene: WheeledSceneCfg = WheeledSceneCfg()

    ## Events
    events: WheeledEventCfg = WheeledEventCfg()

    ## Time
    env_rate: float = 1.0 / 100.0
    agent_rate: float = 1.0 / 25.0

    ## Viewer
    viewer = ViewerCfg(
        eye=(8.0, -8.0, 8.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="env",
    )

    def __post_init__(self):
        super().__post_init__()

        ## Assets -> Scene
        # Robot
        if isinstance(self.robot, AssetVariant):
            # TODO: Implement WheeledRobot from AssetVariant
            raise NotImplementedError()
            self.robot: WheeledRobot = ...
            self.scene.robot = self.robot.asset_cfg
            self.actions = self.robot.action_cfg
        # Terrain
        self.scene.terrain = assets.terrain_from_cfg(
            self,
            seed=self.seed,
            num_assets=1 if self.stack else self.scene.num_envs,
            prim_path="/World/terrain" if self.stack else "{ENV_REGEX_NS}/terrain",
            scale=(64.0, 64.0),
            density=0.2,
            texture_resolution={
                BakeType.ALBEDO: 4096,
                BakeType.NORMAL: 2048,
                BakeType.ROUGHNESS: 1024,
            },
            flat_area_size=4.0,
        )
