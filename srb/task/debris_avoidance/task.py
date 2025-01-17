import sys
from typing import Any, Dict, Sequence, Tuple

import torch
from omni.isaac.lab.managers import EventTermCfg, SceneEntityCfg
from omni.isaac.lab.utils import configclass

import srb.core.envs as env_utils
import srb.core.sim as sim_utils
from srb.asset import Asteroid
from srb.core.asset import Object, RigidObjectCfg
from srb.env import BaseSpacecraftRoboticsEnv, BaseSpacecraftRoboticsEnvCfg, mdp

##############
### Config ###
##############


class ExtDebrisCfg(Object, arbitrary_types_allowed=True):
    ## Model
    asset_cfg: RigidObjectCfg


def asteroid_cfg(
    env_cfg: env_utils.EnvironmentConfig,
    *,
    prim_path: str = "{ENV_REGEX_NS}/sample",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    num_assets: int = 1,
    size: Tuple[float, float] = (0.06, 0.06, 0.04),
    spawn_kwargs: Dict[str, Any] = {},
    procgen_seed_offset: int = 0,
    **kwargs,
) -> ExtDebrisCfg:
    return ExtDebrisCfg(
        asset_cfg=RigidObjectCfg(
            prim_path=prim_path,
            spawn=Asteroid(
                scale=(10, 10, 10),
                num_assets=num_assets,
                seed=env_cfg.seed + procgen_seed_offset,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
                    mesh_approximation="sdf",
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(density=2000.0),
                **spawn_kwargs,
            ),
            **kwargs,
        ),
    )


@configclass
class TaskCfg(BaseSpacecraftRoboticsEnvCfg):
    num_problems_per_env: int = 8

    def __post_init__(self):
        if self.env_cfg.domain != env_utils.Domain.ORBIT:
            print(
                f"[WARN] Environment requires ORBIT scenario ({self.env_cfg.domain} ignored)",
                file=sys.stderr,
            )
            self.env_cfg.domain = env_utils.Domain.ORBIT
        if self.env_cfg.assets.terrain.variant != env_utils.AssetVariant.NONE:
            print(
                f"[WARN] Environment requires NONE terrain ({self.env_cfg.assets.terrain.variant} ignored)",
                file=sys.stderr,
            )
            self.env_cfg.assets.terrain.variant = env_utils.AssetVariant.NONE

        super().__post_init__()

        ## Scene
        self.object_cfgs = [
            asteroid_cfg(
                self.env_cfg,
                prim_path=f"{{ENV_REGEX_NS}}/asteroid{i}",
                asset_cfg=SceneEntityCfg(f"object{i}"),
                num_assets=self.scene.num_envs,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0)),
                procgen_seed_offset=i * self.scene.num_envs,
                spawn_kwargs={
                    "activate_contact_sensors": True,
                },
            )
            for i in range(self.num_problems_per_env)
        ]
        for i, object_cfg in enumerate(self.object_cfgs):
            setattr(self.scene, f"object{i}", object_cfg.asset_cfg)

        ## Events
        self.events.reset_rand_object_state_multi = EventTermCfg(
            func=mdp.reset_root_state_uniform_poisson_disk_3d,
            mode="reset",
            params={
                "asset_cfgs": [
                    SceneEntityCfg(f"object{i}")
                    for i in range(self.num_problems_per_env)
                ],
                "pose_range": {
                    "x": (-5.0, 5.0),
                    "y": (-5.0, 5.0),
                    "z": (-5.0, 5.0),
                    "roll": (-torch.pi, torch.pi),
                    "pitch": (-torch.pi, torch.pi),
                    "yaw": (-torch.pi, torch.pi),
                },
                "velocity_range": {
                    # "x": (-50.0, 50.0),
                    # "y": (-50.0, 50.0),
                    # "z": (-50.0, 50.0),
                    # "roll": (-torch.pi, torch.pi),
                    # "pitch": (-torch.pi, torch.pi),
                    # "yaw": (-torch.pi, torch.pi),
                },
                "radius": (2.0),
            },
        )


############
### Task ###
############


class Task(BaseSpacecraftRoboticsEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Pre-compute metrics used in hot loops
        self._max_episode_length = self.max_episode_length

        ## Initialize the intermediate state
        self._update_intermediate_state()

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: This assumes that `_get_dones()` is called before `_get_rewards()` and `_get_observations()` in `step()`
        self._update_intermediate_state()

        if not self.cfg.enable_truncation:
            self._truncations = torch.zeros_like(self._truncations)

        return self._terminations, self._truncations

    def _get_rewards(self) -> torch.Tensor:
        return self._rewards

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {}

    ########################
    ### Helper Functions ###
    ########################

    def _update_intermediate_state(self):
        ## Compute other intermediate states
        (
            self._remaining_time,
            self._rewards,
            self._terminations,
            self._truncations,
        ) = _compute_intermediate_state(
            current_action=self.action_manager.action,
            previous_action=self.action_manager.prev_action,
            episode_length_buf=self.episode_length_buf,
            max_episode_length=self._max_episode_length,
        )


#############################
### TorchScript functions ###
#############################


@torch.jit.script
def _compute_intermediate_state(
    *,
    current_action: torch.Tensor,
    previous_action: torch.Tensor,
    episode_length_buf: torch.Tensor,
    max_episode_length: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    ## Intermediate states
    # Time
    remaining_time = 1 - (episode_length_buf / max_episode_length).unsqueeze(-1)

    ## Rewards
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -0.05
    penalty_action_rate = WEIGHT_ACTION_RATE * torch.sum(
        torch.square(current_action - previous_action), dim=1
    )

    # Total reward
    rewards = torch.sum(
        torch.stack(
            [
                penalty_action_rate,
            ],
            dim=-1,
        ),
        dim=-1,
    )

    ## Termination and truncation
    truncations = episode_length_buf > (max_episode_length - 1)
    terminations = torch.zeros_like(truncations)

    return (
        remaining_time,
        rewards,
        terminations,
        truncations,
    )
