from typing import Dict, Sequence

import torch

from srb import assets
from srb._typing import IntermediateTaskState
from srb.core.asset import AssetVariant, WheeledRobot
from srb.core.env import WheeledEnv, WheeledEnvCfg, WheeledEventCfg, WheeledSceneCfg
from srb.utils.cfg import configclass
from srb.utils.math import matrix_from_quat, rotmat_to_rot6d, scale_transform

##############
### Config ###
##############


@configclass
class SceneCfg(WheeledSceneCfg):
    pass


@configclass
class EventCfg(WheeledEventCfg):
    pass


@configclass
class TaskCfg(WheeledEnvCfg):
    ## Assets
    robot: WheeledRobot | AssetVariant = assets.Perseverance()

    ## Scene
    scene: SceneCfg = SceneCfg()

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 60.0


############
### Task ###
############


class Task(WheeledEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        assert isinstance(self.cfg.robot, WheeledRobot)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

    def _update_internal_state(self):
        self._internal_state = _compute_internal_state(
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            episode_length_buf=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            robot_joint_pos=self._robot.data.joint_pos,
            robot_quat=self._robot.data.root_quat_w,
            robot_soft_joint_pos_limits=(
                self._robot.data.soft_joint_pos_limits
                if torch.all(torch.isfinite(self._robot.data.soft_joint_pos_limits))
                else None
            ),
            truncate_episodes=self.cfg.truncate_episodes,
        )


@torch.jit.script
def _compute_internal_state(
    *,
    act_current: torch.Tensor,
    act_previous: torch.Tensor,
    episode_length_buf: torch.Tensor,
    max_episode_length: int,
    robot_joint_pos: torch.Tensor,
    robot_quat: torch.Tensor,
    robot_soft_joint_pos_limits: torch.Tensor | None,
    truncate_episodes: bool,
) -> (
    IntermediateTaskState
    | Dict[
        str, torch.Tensor | Dict[str, torch.Tensor] | Dict[str, Dict[str, torch.Tensor]]
    ]
):
    # Robot joints
    if robot_soft_joint_pos_limits is not None:
        joint_pos_normalized = scale_transform(
            robot_joint_pos,
            robot_soft_joint_pos_limits[:, :, 0],
            robot_soft_joint_pos_limits[:, :, 1],
        )
    else:
        joint_pos_normalized = robot_joint_pos

    # Robot pose
    rotmat_robot = matrix_from_quat(robot_quat)
    rot6d_robot = rotmat_to_rot6d(rotmat_robot)

    #############
    ## Rewards ##
    #############
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -0.05
    penalty_action_rate = WEIGHT_ACTION_RATE * torch.sum(
        torch.square(act_current - act_previous), dim=1
    )

    ##################
    ## Terminations ##
    ##################
    termination = torch.zeros(
        episode_length_buf.size(0), dtype=torch.bool, device=episode_length_buf.device
    )
    truncation = (
        episode_length_buf >= max_episode_length
        if truncate_episodes
        else torch.zeros_like(termination)
    )

    return {
        "obs": {
            # "state": {},
            # "state_dyn": {},
            "proprio": {
                "rot6d_robot": rot6d_robot,
            },
            "proprio_dyn": {
                "joint_pos": joint_pos_normalized,
            },
        },
        "rew": {
            "penalty_action_rate": penalty_action_rate,
        },
        "term": termination,
        "trunc": truncation,
    }
