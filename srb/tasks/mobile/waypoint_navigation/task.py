from dataclasses import MISSING
from typing import Sequence, Tuple

import torch

from srb._typing import StepReturn
from srb.core.env import GroundEnv, GroundEnvCfg, GroundEventCfg, GroundSceneCfg
from srb.core.manager import EventTermCfg
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.mdp import offset_pos_natural, randomize_pos
from srb.core.sim import CylinderCfg, PreviewSurfaceCfg
from srb.utils.cfg import configclass

##############
### Config ###
##############


@configclass
class SceneCfg(GroundSceneCfg):
    pass


@configclass
class EventCfg(GroundEventCfg):
    target_pos_random_jump: EventTermCfg = EventTermCfg(
        func=randomize_pos,
        mode="interval",
        interval_range_s=(60.0, 120.0),
        is_global_time=True,
        params={
            "env_attr_name": "_tf_pos_target",
            "pos_range": {
                "x": MISSING,
                "y": MISSING,
            },
        },
    )
    target_pos_evolution: EventTermCfg = EventTermCfg(
        func=offset_pos_natural,
        mode="interval",
        interval_range_s=(0.25, 0.75),
        is_global_time=True,
        params={
            "env_attr_name": "_tf_pos_target",
            "axes": ("x", "y"),
            "step_range": (0.05, 0.5),
            "smoothness": 0.8,
            "pos_bounds": {
                "x": MISSING,
                "y": MISSING,
            },
        },
    )


@configclass
class TaskCfg(GroundEnvCfg):
    ## Scene
    scene: SceneCfg = SceneCfg()
    stack: bool = True

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 120.0
    is_finite_horizon: bool = False

    ## Target
    target_pos_range_ratio: float = 0.9
    tf_quat_target: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target",
        markers={
            "target": CylinderCfg(
                radius=0.01,
                height=20.0,
                visual_material=PreviewSurfaceCfg(emissive_color=(0.2, 0.2, 0.8)),
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()

        # Event: Waypoint target
        assert self.spacing is not None
        for dim in ("x", "y"):
            self.events.target_pos_random_jump.params["pos_range"][dim] = (  # type: ignore
                -0.5 * self.target_pos_range_ratio * self.spacing,
                0.5 * self.target_pos_range_ratio * self.spacing,
            )
            self.events.target_pos_evolution.params["pos_bounds"][dim] = (  # type: ignore
                -0.5 * self.target_pos_range_ratio * self.spacing,
                0.5 * self.target_pos_range_ratio * self.spacing,
            )


############
### Task ###
############


class Task(GroundEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._target_marker: VisualizationMarkers = VisualizationMarkers(
            self.cfg.target_marker_cfg
        )

        ## Initialize buffers
        self._tf_pos_target = self.scene.env_origins + torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._tf_quat_target = torch.tensor(
            self.cfg.tf_quat_target, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

    def extract_step_return(self) -> StepReturn:
        ## Visualize target
        self._target_marker.visualize(self._tf_pos_target, self._tf_quat_target)

        return _compute_step_return(
            ## Time
            episode_length=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            truncate_episodes=self.cfg.truncate_episodes,
            ## Actions
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            ## States
            # Root
            tf_pos_robot=self._robot.data.root_pos_w,
            vel_lin_robot=self._robot.data.root_lin_vel_b,
            vel_ang_robot=self._robot.data.root_ang_vel_b,
            # Transforms (world frame)
            tf_pos_target=self._tf_pos_target,
            # IMU
            imu_lin_acc=self._imu_robot.data.lin_acc_b,
            imu_ang_vel=self._imu_robot.data.ang_vel_b,
        )


@torch.jit.script
def _compute_step_return(
    *,
    ## Time
    episode_length: torch.Tensor,
    max_episode_length: int,
    truncate_episodes: bool,
    ## Actions
    act_current: torch.Tensor,
    act_previous: torch.Tensor,
    ## States
    # Root
    tf_pos_robot: torch.Tensor,
    vel_lin_robot: torch.Tensor,
    vel_ang_robot: torch.Tensor,
    # Transforms (world frame)
    tf_pos_target: torch.Tensor,
    # IMU
    imu_lin_acc: torch.Tensor,
    imu_ang_vel: torch.Tensor,
) -> StepReturn:
    num_envs = episode_length.size(0)
    # dtype = episode_length.dtype
    device = episode_length.device

    ############
    ## States ##
    ############
    ## Transforms (world frame)
    # Robot -> Target
    tf_pos_robot_to_target = tf_pos_robot - tf_pos_target
    dist_robot_to_target = torch.norm(tf_pos_robot_to_target, dim=-1)

    #############
    ## Rewards ##
    #############
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -0.05
    penalty_action_rate = WEIGHT_ACTION_RATE * torch.sum(
        torch.square(act_current - act_previous), dim=1
    )

    # Penalty: Angular velocity
    WEIGHT_ANGULAR_VELOCITY = -0.1
    penalty_angular_velocity = WEIGHT_ANGULAR_VELOCITY * torch.norm(
        vel_ang_robot, dim=-1
    )

    # Penalty: Distance | Robot <--> Target
    WEIGHT_DISTANCE_ROBOT_TO_TARGET = -16.0
    penalty_distance_robot_to_target = (
        WEIGHT_DISTANCE_ROBOT_TO_TARGET * dist_robot_to_target
    )

    # Reward: Distance | Robot <--> Target (precision)
    WEIGHT_DISTANCE_ROBOT_TO_TARGET_PRECISION = 32.0
    TANH_STD_DISTANCE_ROBOT_TO_TARGET_PRECISION = 0.05
    reward_distance_robot_to_target_precision = (
        WEIGHT_DISTANCE_ROBOT_TO_TARGET_PRECISION
        * (
            1.0
            - torch.tanh(
                dist_robot_to_target / TANH_STD_DISTANCE_ROBOT_TO_TARGET_PRECISION
            )
        )
    )

    ##################
    ## Terminations ##
    ##################
    # No termination condition
    termination = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # Truncation
    truncation = (
        episode_length >= max_episode_length
        if truncate_episodes
        else torch.zeros(num_envs, dtype=torch.bool, device=device)
    )

    return StepReturn(
        {
            "state": {
                "vel_lin_robot": vel_lin_robot,
                "vel_ang_robot": vel_ang_robot,
                "tf_pos_robot_to_target": tf_pos_robot_to_target,
            },
            # "state_dyn": {},
            "proprio": {
                "imu_lin_acc": imu_lin_acc,
                "imu_ang_vel": imu_ang_vel,
            },
            # "proprio_dyn": {},
        },
        {
            "penalty_action_rate": penalty_action_rate,
            "penalty_angular_velocity": penalty_angular_velocity,
            "penalty_distance_robot_to_target": penalty_distance_robot_to_target,
            "reward_distance_robot_to_target_precision": reward_distance_robot_to_target_precision,
        },
        termination,
        truncation,
    )
