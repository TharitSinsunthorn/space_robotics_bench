from dataclasses import MISSING
from typing import Dict, List, Sequence

import torch

from srb._typing import IntermediateTaskState
from srb.core.asset import (
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    SingleArmManipulator,
)
from srb.core.env import SingleArmEnv
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import VisualizationMarkers
from srb.core.mdp import reset_collection_root_state_uniform_poisson_disk_2d
from srb.core.sensor import ContactSensor
from srb.utils.cfg import configclass
from srb.utils.math import (
    matrix_from_quat,
    rotmat_to_rot6d,
    scale_transform,
    subtract_frame_transforms,
)

from .asset import sample_cfg
from .task import EventCfg, SceneCfg, TaskCfg

##############
### Config ###
##############


@configclass
class MultiSceneCfg(SceneCfg):
    obj: None = None
    objs: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects=MISSING,  # type: ignore
    )


@configclass
class MultiEventCfg(EventCfg):
    randomize_object_state: EventTermCfg = EventTermCfg(
        func=reset_collection_root_state_uniform_poisson_disk_2d,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("objs"),
            "pose_range": MISSING,
            "velocity_range": MISSING,
            "radius": 0.1,
        },
    )


@configclass
class MultiTaskCfg(TaskCfg):
    ## Scene
    scene: MultiSceneCfg = MultiSceneCfg()
    num_problems_per_env: int = 8

    ## Events
    events: MultiEventCfg = MultiEventCfg()

    ## Time
    episode_length_s: float = MISSING  # type: ignore
    _base_episode_length_s: float = 7.5

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.robot, SingleArmManipulator)

        ## Time
        self.episode_length_s = self.num_problems_per_env * self._base_episode_length_s

        ## Assets -> Scene
        # Object
        self.scene.obj = None
        objs = [
            sample_cfg(
                self,
                seed=self.seed + (i * self.scene.num_envs),
                num_assets=self.scene.num_envs,
                prim_path=f"{{ENV_REGEX_NS}}/sample{i}",
                asset_cfg=SceneEntityCfg(f"obj{i}"),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0)),
                activate_contact_sensors=True,
            )
            for i in range(self.num_problems_per_env)
        ]
        self.scene.objs.rigid_objects = {
            f"obj{i}": cfg.asset_cfg for i, cfg in enumerate(objs)
        }
        # Sensor: Contacts | Robot hand <--> Object
        self.scene.contacts_robot_hand_obj.filter_prim_paths_expr = [
            f"{{ENV_REGEX_NS}}/sample{i}" for i in range(self.num_problems_per_env)
        ]

        ## Events
        self.events.randomize_object_state.params["pose_range"] = objs[
            0
        ].state_randomizer.params["pose_range"]
        self.events.randomize_object_state.params["velocity_range"] = objs[
            0
        ].state_randomizer.params["velocity_range"]


############
### Task ###
############


class MultiTask(SingleArmEnv):
    cfg: MultiTaskCfg

    def __init__(self, cfg: MultiTaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        assert isinstance(self.cfg.robot, SingleArmManipulator)

        ## Get scene assets
        self._contacts_robot_hand_obj: ContactSensor = self.scene[
            "contacts_robot_hand_obj"
        ]
        self._objs: RigidObjectCollection = self.scene["objs"]
        self._target_marker: VisualizationMarkers = VisualizationMarkers(
            self.cfg.target_marker_cfg
        )

        ## Initialize buffers
        self._obj_initial_pos_z = torch.zeros(
            (self.num_envs, self.cfg.num_problems_per_env),
            dtype=torch.float32,
            device=self.device,
        )
        self._target_pos = self.scene.env_origins.unsqueeze(1) + torch.tensor(
            self.cfg.target_pos, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, self.cfg.num_problems_per_env, 1)
        self._target_quat = torch.tensor(
            self.cfg.target_quat, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, self.cfg.num_problems_per_env, 1)

        ## Cache metrics
        self._robot_joint_indices_arm, _ = self._robot.find_joints(
            self.cfg.robot.regex_joints_arm
        )
        self._robot_joint_indices_hand, _ = self._robot.find_joints(
            self.cfg.robot.regex_joints_hand
        )

        ## Visualize target
        self._target_marker.visualize(
            self._target_pos[:, 0, :], self._target_quat[:, 0, :]
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._obj_initial_pos_z[env_ids] = self._objs.data.object_pos_w[env_ids, :, 2]

    def _update_internal_state(self):
        self._internal_state = _compute_internal_state(
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            episode_length_buf=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            obj_initial_pos_z=self._obj_initial_pos_z,
            obj_pos=self._objs.data.object_com_pos_w,
            obj_quat=self._objs.data.object_com_quat_w,
            robot_contact_forces_arm=self._contacts_robot_arm.data.net_forces_w,  # type: ignore
            robot_contact_forces_hand_matrix=self._contacts_robot_hand_obj.data.force_matrix_w,  # type: ignore
            robot_ee_pos_wrt_base=self._tf_robot_ee.data.target_pos_source[:, 0, :],
            robot_ee_pos=self._tf_robot_ee.data.target_pos_w[:, 0, :],
            robot_ee_quat_wrt_base=self._tf_robot_ee.data.target_quat_source[:, 0, :],
            robot_ee_quat=self._tf_robot_ee.data.target_quat_w[:, 0, :],
            robot_incoming_forces=self._robot.root_physx_view.get_link_incoming_joint_force(),
            robot_joint_indices_arm=self._robot_joint_indices_arm,
            robot_joint_indices_hand=self._robot_joint_indices_hand,
            robot_joint_pos=self._robot.data.joint_pos,
            robot_soft_joint_pos_limits=self._robot.data.soft_joint_pos_limits,
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
    obj_initial_pos_z: torch.Tensor,
    obj_pos: torch.Tensor,
    obj_quat: torch.Tensor,
    robot_contact_forces_arm: torch.Tensor,
    robot_contact_forces_hand_matrix: torch.Tensor,
    robot_ee_pos_wrt_base: torch.Tensor,
    robot_ee_pos: torch.Tensor,
    robot_ee_quat_wrt_base: torch.Tensor,
    robot_ee_quat: torch.Tensor,
    robot_incoming_forces: torch.Tensor,
    robot_joint_indices_arm: List[int],
    robot_joint_indices_hand: List[int],
    robot_joint_pos: torch.Tensor,
    robot_soft_joint_pos_limits: torch.Tensor,
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

    # Robot joints
    joint_pos_normalized = scale_transform(
        robot_joint_pos,
        robot_soft_joint_pos_limits[:, :, 0],
        robot_soft_joint_pos_limits[:, :, 1],
    )
    joint_pos_arm_normalized, joint_pos_hand_normalized = (
        joint_pos_normalized[:, robot_joint_indices_arm],
        joint_pos_normalized[:, robot_joint_indices_hand],
    )
    joint_pos_hand_normalized_mean = joint_pos_hand_normalized.mean(
        dim=-1, keepdim=True
    )

    # Robot base -> End-effector
    rotmat_robot_base_to_robot_ee = matrix_from_quat(robot_ee_quat_wrt_base)
    rot6d_robot_base_to_robot_ee = rotmat_to_rot6d(rotmat_robot_base_to_robot_ee)

    # End-effector -> Object
    pos_robot_ee_to_obj, quat_robot_ee_to_obj = subtract_frame_transforms(
        t01=robot_ee_pos.unsqueeze(1).repeat(1, num_problems_per_env, 1),
        q01=robot_ee_quat.unsqueeze(1).repeat(1, num_problems_per_env, 1),
        t02=obj_pos,
        q02=obj_quat,
    )
    rotmat_robot_ee_to_obj = matrix_from_quat(quat_robot_ee_to_obj)
    rot6d_robot_ee_to_obj = rotmat_to_rot6d(rotmat_robot_ee_to_obj)

    # Object -> Target
    pos_obj_to_target, quat_obj_to_target = subtract_frame_transforms(
        t01=obj_pos,
        q01=obj_quat,
        t02=target_pos,
        q02=target_quat,
    )
    rotmat_obj_to_target = matrix_from_quat(quat_obj_to_target)
    rot6d_obj_to_target = rotmat_to_rot6d(rotmat_obj_to_target)

    # Robot hand wrench
    wrench_robot_hand = robot_incoming_forces[:, robot_joint_indices_hand]
    wrench_robot_hand_mean = wrench_robot_hand.mean(dim=1)

    #############
    ## Rewards ##
    #############
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -0.05
    penalty_action_rate = WEIGHT_ACTION_RATE * torch.sum(
        torch.square(act_current - act_previous), dim=1
    )

    # Penalty: Undesired robot arm contacts
    WEIGHT_UNDERSIRED_ROBOT_ARM_CONTACTS = -0.1
    THRESHOLD_UNDERSIRED_ROBOT_ARM_CONTACTS = 10.0
    penalty_undersired_robot_arm_contacts = WEIGHT_UNDERSIRED_ROBOT_ARM_CONTACTS * (
        torch.max(torch.norm(robot_contact_forces_arm, dim=-1), dim=1)[0]
        > THRESHOLD_UNDERSIRED_ROBOT_ARM_CONTACTS
    )

    # Reward: Distance | End-effector <--> Object
    WEIGHT_DISTANCE_EE_TO_OBJ = 1.0
    TANH_STD_DISTANCE_EE_TO_OBJ = 0.25
    reward_distance_ee_to_obj = WEIGHT_DISTANCE_EE_TO_OBJ * (
        1.0
        - torch.tanh(
            torch.norm(pos_robot_ee_to_obj, dim=-1) / TANH_STD_DISTANCE_EE_TO_OBJ
        )
    ).sum(dim=1)

    # Reward: Object grasped
    WEIGHT_OBJ_GRASPED = 4.0
    THRESHOLD_OBJ_GRASPED = 5.0
    reward_obj_grasped = WEIGHT_OBJ_GRASPED * (
        torch.mean(
            torch.max(torch.norm(robot_contact_forces_hand_matrix, dim=-1), dim=-1)[0],
            dim=1,
        )
        > THRESHOLD_OBJ_GRASPED
    )

    # Reward: Object lifted
    WEIGHT_OBJ_LIFTED = 16.0
    HEIGHT_OFFSET_OBJ_LIFTED = 0.5
    HEIGHT_SPAN_OBJ_LIFTED = 0.25
    TAHN_STD_HEIGHT_OBJ_LIFTED = 0.1
    obj_target_height_offset = (
        torch.abs(obj_pos[:, :, 2] - obj_initial_pos_z - HEIGHT_OFFSET_OBJ_LIFTED)
        - HEIGHT_SPAN_OBJ_LIFTED
    ).clamp(min=0.0)
    reward_obj_lifted = WEIGHT_OBJ_LIFTED * (
        1.0 - torch.tanh(obj_target_height_offset / TAHN_STD_HEIGHT_OBJ_LIFTED)
    ).sum(dim=1)

    # Reward: Distance | Object <--> Target
    WEIGHT_DISTANCE_OBJ_TO_TARGET = 32.0
    TANH_STD_DISTANCE_OBJ_TO_TARGET = 0.333
    reward_distance_obj_to_target = WEIGHT_DISTANCE_OBJ_TO_TARGET * (
        1.0
        - torch.tanh(
            torch.norm(pos_obj_to_target, dim=-1) / TANH_STD_DISTANCE_OBJ_TO_TARGET
        )
    ).sum(dim=1)

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
                "pos_obj_to_target": pos_obj_to_target,
                "pos_robot_ee_to_obj": pos_robot_ee_to_obj,
                "rot6d_obj_to_target": rot6d_obj_to_target,
                "rot6d_robot_ee_to_obj": rot6d_robot_ee_to_obj,
                "wrench_robot_hand_mean": wrench_robot_hand_mean,
            },
            "state_dyn": {
                "wrench_robot_hand": wrench_robot_hand,
            },
            "proprio": {
                "joint_pos_hand_normalized_mean": joint_pos_hand_normalized_mean,
                "remaining_time": remaining_time,
                "robot_ee_pos_wrt_base": robot_ee_pos_wrt_base,
                "rot6d_robot_base_to_robot_ee": rot6d_robot_base_to_robot_ee,
            },
            "proprio_dyn": {
                "joint_pos_arm_normalized": joint_pos_arm_normalized,
                "joint_pos_hand_normalized": joint_pos_hand_normalized,
            },
        },
        "rew": {
            "penalty_action_rate": penalty_action_rate,
            "penalty_undersired_robot_arm_contacts": penalty_undersired_robot_arm_contacts,
            "reward_distance_ee_to_obj": reward_distance_ee_to_obj,
            "reward_distance_obj_to_target": reward_distance_obj_to_target,
            "reward_obj_grasped": reward_obj_grasped,
            "reward_obj_lifted": reward_obj_lifted,
        },
        "term": termination,
        "trunc": truncation,
    }
