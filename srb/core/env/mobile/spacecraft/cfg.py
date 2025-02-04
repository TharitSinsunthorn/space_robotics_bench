import torch

from srb import assets
from srb.core.asset import AssetVariant, Spacecraft, Terrain
from srb.core.env import BaseEventCfg, BaseSceneCfg, DirectEnvCfg, Domain, ViewerCfg
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import reset_root_state_uniform
from srb.core.sim import SimforgeAssetCfg
from srb.utils.cfg import configclass


@configclass
class SpacecraftSceneCfg(BaseSceneCfg):
    pass


@configclass
class SpacecraftEventCfg(BaseEventCfg):
    randomize_robot_state: EventTermCfg | None = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (
                    -torch.pi,
                    torch.pi,
                ),
                "pitch": (
                    -torch.pi,
                    torch.pi,
                ),
                "yaw": (
                    -torch.pi,
                    torch.pi,
                ),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.7854, 0.7854),
                "pitch": (-0.7854, 0.7854),
                "yaw": (-0.7854, 0.7854),
            },
        },
    )


@configclass
class SpacecraftEnvCfg(DirectEnvCfg):
    ## Scenario
    domain: Domain = Domain.ORBIT

    ## Assets
    robot: Spacecraft | AssetVariant = assets.Cubesat()
    terrain: Terrain | AssetVariant | None = None

    ## Scene
    scene: SpacecraftSceneCfg = SpacecraftSceneCfg(env_spacing=2.0)

    ## Events
    events: SpacecraftEventCfg = SpacecraftEventCfg()

    ## Time
    env_rate: float = 1.0 / 50.0
    agent_rate: float = 1.0 / 10.0

    ## Viewer
    viewer = ViewerCfg(
        eye=(16.0, -16.0, 16.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="env",
    )

    def __post_init__(self):
        super().__post_init__()

        ## Assets -> Scene
        # Robot
        if isinstance(self.robot, AssetVariant):
            # TODO: Implement Spacecraft from AssetVariant
            raise NotImplementedError()
            self.robot: Spacecraft = ...
            self.scene.robot = self.robot.asset_cfg
            self.actions = self.robot.action_cfg
        if isinstance(self.robot.asset_cfg.spawn, SimforgeAssetCfg):
            # TODO: Set the number of procedural variants in a better way
            self.robot.asset_cfg.spawn.num_assets = self.scene.num_envs
            self.robot.asset_cfg.spawn.seed = self.seed
            self.scene.robot = self.robot.asset_cfg
