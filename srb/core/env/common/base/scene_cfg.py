from dataclasses import MISSING

from srb.core.asset import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from srb.core.env import InteractiveSceneCfg
from srb.utils.cfg import configclass


@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    env_spacing: float = 0.0
    replicate_physics: bool = False

    light: AssetBaseCfg | None = None
    sky: AssetBaseCfg | None = None

    robot: ArticulationCfg | RigidObjectCfg = MISSING  # type: ignore
    terrain: AssetBaseCfg | None = None
    obj: AssetBaseCfg | ArticulationCfg | RigidObjectCfg | None = None
