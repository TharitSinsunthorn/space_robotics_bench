from dataclasses import MISSING
from typing import Dict, Sequence, Tuple

import torch
from simforge import TexResConfig

from srb import assets
from srb.core.asset import RigidObjectCfg
from srb.core.env import SpacecraftEnv, SpacecraftEnvCfg, SpacecraftEventCfg
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import reset_root_state_uniform_poisson_disk_3d
from srb.utils.cfg import configclass

##############
### Config ###
##############


@configclass
class EventCfg(SpacecraftEventCfg):
    ## Object
    randomize_object_state: EventTermCfg | None = EventTermCfg(
        func=reset_root_state_uniform_poisson_disk_3d,
        mode="reset",
        params={
            "asset_cfg": MISSING,
            "pose_range": {
                "x": (-5.0, 5.0),
                "y": (-5.0, 5.0),
                "z": (-5.0, 5.0),
                "roll": (-torch.pi, torch.pi),
                "pitch": (-torch.pi, torch.pi),
                "yaw": (-torch.pi, torch.pi),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1 * torch.pi, 0.1 * torch.pi),
                "pitch": (-0.1 * torch.pi, 0.1 * torch.pi),
                "yaw": (-0.1 * torch.pi, 0.1 * torch.pi),
            },
            "radius": (2.0),
        },
    )


def debris_cfg(
    *,
    seed: int,
    num_assets: int,
    prim_path: str = "{ENV_REGEX_NS}/object",
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    texture_resolution: TexResConfig | None = None,
    **kwargs,
) -> RigidObjectCfg:
    asset_cfg = assets.Asteroid(
        scale=scale, texture_resolution=texture_resolution
    ).asset_cfg

    asset_cfg.spawn.seed = seed  # type: ignore
    asset_cfg.spawn.num_assets = num_assets  # type: ignore
    asset_cfg.prim_path = prim_path
    asset_cfg.spawn.replace(**kwargs)

    return asset_cfg


@configclass
class TaskCfg(SpacecraftEnvCfg):
    num_problems_per_env: int = 8

    ## Environment
    episode_length_s: float = 20.0

    ## Task
    is_finite_horizon: bool = False

    ## Events
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        ## Scene
        self.objects = [
            debris_cfg(
                prim_path=f"{{ENV_REGEX_NS}}/debris{i}",
                seed=self.seed + (i * self.scene.num_envs),
                num_assets=self.scene.num_envs,
                activate_contact_sensors=True,
            )
            for i in range(self.num_problems_per_env)
        ]
        for i, obj_cfg in enumerate(self.objects):
            setattr(self.scene, f"object{i}", obj_cfg)

        ## Events
        self.events.randomize_object_state.params["asset_cfg"] = [  # type: ignore
            SceneEntityCfg(f"object{i}") for i in range(self.num_problems_per_env)
        ]


############
### Task ###
############


class Task(SpacecraftEnv):
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
