from dataclasses import MISSING
from typing import Dict, Sequence, Tuple

import torch

from srb import assets
from srb._typing import IntermediateTaskState
from srb.core.asset import (
    AssetVariant,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    Spacecraft,
)
from srb.core.env import (
    SpacecraftEnv,
    SpacecraftEnvCfg,
    SpacecraftEventCfg,
    SpacecraftSceneCfg,
)
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.mdp import reset_collection_root_state_uniform_poisson_disk_3d
from srb.core.sim import PreviewSurfaceCfg, SphereCfg
from srb.utils.cfg import configclass
from srb.utils.math import matrix_from_quat, rotmat_to_rot6d, subtract_frame_transforms

from .asset import debris_cfg

##############
### Config ###
##############


@configclass
class SceneCfg(SpacecraftSceneCfg):
    env_spacing = 0.0  # NOTE: Needs to be 0.0 due to target position (can be improved)

    objs: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects=MISSING,  # type: ignore
    )


@configclass
class EventCfg(SpacecraftEventCfg):
    randomize_object_state: EventTermCfg = EventTermCfg(
        func=reset_collection_root_state_uniform_poisson_disk_3d,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("objs"),
            "pose_range": {
                "x": (-10.0, 10.0),
                "y": (-10.0, 10.0),
                "z": (-50.0, 10.0),
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
            "radius": (5.0),
        },
    )


@configclass
class TaskCfg(SpacecraftEnvCfg):
    ## Assets
    robot: Spacecraft | AssetVariant = assets.Cubesat()

    ## Scene
    scene: SceneCfg = SceneCfg()
    num_problems_per_env: int = 20

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 30.0
    is_finite_horizon: bool = False

    ## Target
    target_pos: Tuple[float, float, float] = (0.0, 0.0, -50.0)
    target_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target",
        markers={
            "target": SphereCfg(
                visible=False,
                radius=0.025,
                visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.robot, Spacecraft)

        ## Assets -> Scene
        # Object
        self.scene.objs.rigid_objects = {
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

        ## Get scene assets
        self._objs: RigidObjectCollection = self.scene["objs"]
        self._target_marker: VisualizationMarkers = VisualizationMarkers(
            self.cfg.target_marker_cfg
        )

        ## Initialize buffers
        self._target_pos = self.scene.env_origins + torch.tensor(
            self.cfg.target_pos, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._target_quat = torch.tensor(
            self.cfg.target_quat, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)

        ## Visualize target
        self._target_marker.visualize(self._target_pos, self._target_quat)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

    def _update_internal_state(self):
        self._internal_state = _compute_internal_state(
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            episode_length_buf=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            obj_pos=self._objs.data.object_com_pos_w,
            obj_quat=self._objs.data.object_com_quat_w,
            robot_pos=self._robot.data.root_com_pos_w,
            robot_quat=self._robot.data.root_com_quat_w,
            target_pos=self._target_pos,
            target_quat=self._target_quat,
            truncate_episodes=self.cfg.truncate_episodes,
        )


@torch.jit.script
def _compute_internal_state(
    *,
    act_current: torch.Tensor,
    act_previous: torch.Tensor,
    episode_length_buf: torch.Tensor,
    max_episode_length: int,
    obj_pos: torch.Tensor,
    obj_quat: torch.Tensor,
    robot_pos: torch.Tensor,
    robot_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    truncate_episodes: bool,
) -> (
    IntermediateTaskState
    | Dict[
        str, torch.Tensor | Dict[str, torch.Tensor] | Dict[str, Dict[str, torch.Tensor]]
    ]
):
    num_problems_per_env = obj_pos.size(1)

    # Time
    remaining_time = 1 - (episode_length_buf / max_episode_length).unsqueeze(-1)

    # Robot -> Object
    pos_robot_to_obj, quat_robot_to_obj = subtract_frame_transforms(
        t01=robot_pos.unsqueeze(1).repeat(1, num_problems_per_env, 1),
        q01=robot_quat.unsqueeze(1).repeat(1, num_problems_per_env, 1),
        t02=obj_pos,
        q02=obj_quat,
    )
    rotmat_robot_to_obj = matrix_from_quat(quat_robot_to_obj)
    rot6d_robot_to_obj = rotmat_to_rot6d(rotmat_robot_to_obj)
    nearest_obj_idx = torch.argmin(torch.norm(pos_robot_to_obj, dim=-1), dim=1)
    pos_robot_to_obj_nearest = pos_robot_to_obj[
        torch.arange(pos_robot_to_obj.size(0)), nearest_obj_idx
    ]
    rot6d_robot_to_obj_nearest = rot6d_robot_to_obj[
        torch.arange(rot6d_robot_to_obj.size(0)), nearest_obj_idx
    ]

    # Robot -> Target
    pos_robot_to_target, quat_robot_to_target = subtract_frame_transforms(
        t01=robot_pos,
        q01=robot_quat,
        t02=target_pos,
        q02=target_quat,
    )
    rotmat_robot_to_target = matrix_from_quat(quat_robot_to_target)
    rot6d_robot_to_target = rotmat_to_rot6d(rotmat_robot_to_target)

    #############
    ## Rewards ##
    #############
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -0.05
    penalty_action_rate = WEIGHT_ACTION_RATE * torch.sum(
        torch.square(act_current - act_previous), dim=1
    )

    # Reward: Distance | Robot <--> Object
    WEIGHT_DISTANCE_ROBOT_TO_OBJ = 1.0
    reward_distance_robot_to_obj = WEIGHT_DISTANCE_ROBOT_TO_OBJ * torch.norm(
        pos_robot_to_obj_nearest, dim=-1
    )

    # Penalty: Distance | Robot <--> Target
    WEIGHT_DISTANCE_ROBOT_TO_TARGET = -32.0
    penalty_distance_robot_to_target = WEIGHT_DISTANCE_ROBOT_TO_TARGET * torch.norm(
        pos_robot_to_target, dim=-1
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
            "state": {
                "pos_robot_to_obj_nearest": pos_robot_to_obj_nearest,
                "rot6d_robot_to_obj_nearest": rot6d_robot_to_obj_nearest,
                "pos_robot_to_target": pos_robot_to_target,
                "rot6d_robot_to_target": rot6d_robot_to_target,
            },
            "state_dyn": {
                "pos_robot_to_obj": pos_robot_to_obj,
                "rot6d_robot_to_obj": rot6d_robot_to_obj,
            },
            "proprio": {
                "remaining_time": remaining_time,
            },
            # "proprio_dyn": {},
        },
        "rew": {
            "penalty_action_rate": penalty_action_rate,
            "reward_distance_robot_to_obj": reward_distance_robot_to_obj,
            "penalty_distance_robot_to_target": penalty_distance_robot_to_target,
        },
        "term": termination,
        "trunc": truncation,
    }
