from dataclasses import MISSING
from typing import Dict, Sequence, Tuple

import torch

from srb import assets
from srb.core.asset import AssetVariant, RigidObjectCollectionCfg, Spacecraft
from srb.core.env import (
    SpacecraftEnv,
    SpacecraftEnvCfg,
    SpacecraftEventCfg,
    SpacecraftSceneCfg,
)
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import reset_collection_root_state_uniform_poisson_disk_3d
from srb.utils.cfg import configclass

from .asset import debris_cfg

##############
### Config ###
##############


@configclass
class SceneCfg(SpacecraftSceneCfg):
    objects: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects=MISSING,  # type: ignore
    )


@configclass
class EventCfg(SpacecraftEventCfg):
    randomize_object_state: EventTermCfg = EventTermCfg(
        func=reset_collection_root_state_uniform_poisson_disk_3d,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("objects"),
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


@configclass
class TaskCfg(SpacecraftEnvCfg):
    ## Assets
    robot: Spacecraft | AssetVariant = assets.Cubesat()

    ## Scene
    scene: SceneCfg = SceneCfg()
    num_problems_per_env: int = 8

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 30.0
    is_finite_horizon: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.robot, Spacecraft)

        ## Assets -> Scene
        # Object
        self.scene.objects.rigid_objects = {
            f"obj{i}": debris_cfg(
                prim_path=f"{{ENV_REGEX_NS}}/debris{i}",
                seed=self.seed + (i * self.scene.num_envs),
                num_assets=self.scene.num_envs,
                activate_contact_sensors=True,
            )
            for i in range(self.num_problems_per_env)
        }


############
### Task ###
############


class Task(SpacecraftEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        assert isinstance(self.cfg.robot, Spacecraft)

        ## Pre-compute metrics used in hot loops
        self._max_episode_length = self.max_episode_length

        ## Initialize the intermediate state
        self._update_intermediate_state()

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_intermediate_state()

        if not self.cfg.enable_truncation:
            self._truncations = torch.zeros_like(self._truncations)

        return self._terminations, self._truncations

    def _get_rewards(self) -> torch.Tensor:
        return self._rewards

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {}

    def _update_intermediate_state(self):
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
