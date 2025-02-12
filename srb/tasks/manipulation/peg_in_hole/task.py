from dataclasses import MISSING
from typing import Dict, List, Sequence

import torch

from srb._typing import IntermediateTaskState
from srb.core.asset import RigidObject, RigidObjectCfg, SingleArmManipulator
from srb.core.env import (
    SingleArmEnv,
    SingleArmEnvCfg,
    SingleArmEventCfg,
    SingleArmSceneCfg,
)
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import reset_root_state_uniform
from srb.core.sensor import ContactSensor, ContactSensorCfg
from srb.utils.cfg import configclass
from srb.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    rotmat_to_rot6d,
    scale_transform,
    subtract_frame_transforms,
)

from .asset import peg_and_hole_cfg

##############
### Config ###
##############


@configclass
class SceneCfg(SingleArmSceneCfg):
    obj: RigidObjectCfg = MISSING  # type: ignore
    target: RigidObjectCfg = MISSING  # type: ignore
    contacts_robot_hand_obj: ContactSensorCfg = ContactSensorCfg(
        prim_path=MISSING,  # type: ignore
        filter_prim_paths_expr=MISSING,  # type: ignore
    )


@configclass
class EventCfg(SingleArmEventCfg):
    randomize_object_state: EventTermCfg = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("obj"),
            "pose_range": {
                "x": (-0.25 - 0.025, -0.25 + 0.0125),
                "y": (-0.05, 0.05),
                "roll": (torch.pi / 2, torch.pi / 2),
                "yaw": (
                    torch.pi / 2 - torch.pi / 16,
                    torch.pi / 2 + torch.pi / 16,
                ),
            },
            "velocity_range": {},
        },
    )


@configclass
class TaskCfg(SingleArmEnvCfg):
    ## Scene
    scene: SceneCfg = SceneCfg()

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 10.0
    is_finite_horizon: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.robot, SingleArmManipulator)

        ## Assets -> Scene
        # Object + Target
        self.problem_cfg = peg_and_hole_cfg(
            self,
            seed=self.seed,
            num_assets=self.scene.num_envs,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.02)),
            peg_kwargs={
                "activate_contact_sensors": True,
            },
        )
        # Object
        self.scene.obj = self.problem_cfg.peg.asset_cfg
        # Target
        self.scene.target = self.problem_cfg.hole.asset_cfg
        # Sensor: Contacts | Robot hand <--> Object
        self.scene.contacts_robot_hand_obj.prim_path = (
            f"{self.scene.robot.prim_path}/{self.robot.regex_links_hand}"
        )
        self.scene.contacts_robot_hand_obj.filter_prim_paths_expr = [
            self.scene.obj.prim_path
        ]


############
### Task ###
############


class Task(SingleArmEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        assert isinstance(self.cfg.robot, SingleArmManipulator)

        ## Get scene assets
        self._contacts_robot_hand_obj: ContactSensor = self.scene[
            "contacts_robot_hand_obj"
        ]
        self._obj: RigidObject = self.scene["obj"]
        self._target: RigidObject = self.scene["target"]

        ## Initialize buffers
        self._obj_initial_pos_z = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._peg_offset_pos_ends = torch.tensor(
            self.cfg.problem_cfg.peg.offset_pos_ends,
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1, 1)
        self._peg_rot_symmetry_n = torch.tensor(
            self.cfg.problem_cfg.peg.rot_symmetry_n,
            dtype=torch.int32,
            device=self.device,
        ).repeat(self.num_envs)
        self._hole_offset_pos_bottom = torch.tensor(
            self.cfg.problem_cfg.hole.offset_pos_bottom,
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self._hole_offset_pos_entrance = torch.tensor(
            self.cfg.problem_cfg.hole.offset_pos_entrance,
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)

        ## Cache metrics
        self._robot_joint_indices_arm, _ = self._robot.find_joints(
            self.cfg.robot.regex_joints_arm
        )
        self._robot_joint_indices_hand, _ = (
            self._robot.find_joints(self.cfg.robot.regex_joints_hand)
            if self.cfg.robot.regex_joints_hand
            else ([], [])
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._obj_initial_pos_z[env_ids] = self._obj.data.root_pos_w[env_ids, 2]

    def _update_internal_state(self):
        self._internal_state = _compute_internal_state(
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            episode_length_buf=self.episode_length_buf,
            hole_offset_pos_bottom=self._hole_offset_pos_bottom,
            hole_offset_pos_entrance=self._hole_offset_pos_entrance,
            max_episode_length=self.max_episode_length,
            obj_initial_pos_z=self._obj_initial_pos_z,
            obj_pos=self._obj.data.root_pos_w,
            obj_quat=self._obj.data.root_quat_w,
            peg_offset_pos_ends=self._peg_offset_pos_ends,
            peg_rot_symmetry_n=self._peg_rot_symmetry_n,
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
            robot_soft_joint_pos_limits=(
                self._robot.data.soft_joint_pos_limits
                if torch.all(torch.isfinite(self._robot.data.soft_joint_pos_limits))
                else None
            ),
            target_pos=self._target.data.root_pos_w,
            target_quat=self._target.data.root_quat_w,
            truncate_episodes=self.cfg.truncate_episodes,
        )


@torch.jit.script
def _compute_internal_state(
    *,
    act_current: torch.Tensor,
    act_previous: torch.Tensor,
    episode_length_buf: torch.Tensor,
    hole_offset_pos_bottom: torch.Tensor,
    hole_offset_pos_entrance: torch.Tensor,
    max_episode_length: int,
    obj_initial_pos_z: torch.Tensor,
    obj_pos: torch.Tensor,
    obj_quat: torch.Tensor,
    peg_offset_pos_ends: torch.Tensor,
    peg_rot_symmetry_n: torch.Tensor,
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
    robot_soft_joint_pos_limits: torch.Tensor | None,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    truncate_episodes: bool,
) -> (
    IntermediateTaskState
    | Dict[
        str, torch.Tensor | Dict[str, torch.Tensor] | Dict[str, Dict[str, torch.Tensor]]
    ]
):
    # Time
    remaining_time = 1 - (episode_length_buf / max_episode_length).unsqueeze(-1)

    # Robot joints
    if robot_soft_joint_pos_limits is not None:
        joint_pos_normalized = scale_transform(
            robot_joint_pos,
            robot_soft_joint_pos_limits[:, :, 0],
            robot_soft_joint_pos_limits[:, :, 1],
        )
    else:
        joint_pos_normalized = robot_joint_pos
    joint_pos_arm_normalized, joint_pos_hand_normalized = (
        joint_pos_normalized[:, robot_joint_indices_arm],
        joint_pos_normalized[:, robot_joint_indices_hand],
    )
    joint_pos_hand_normalized_mean = (
        joint_pos_hand_normalized.mean(dim=-1, keepdim=True)
        if torch.numel(joint_pos_hand_normalized)
        else torch.zeros(
            (joint_pos_arm_normalized.shape[0], 1),
            dtype=joint_pos_arm_normalized.dtype,
            device=joint_pos_arm_normalized.device,
        )
    )

    # Robot base -> End-effector
    rotmat_robot_base_to_robot_ee = matrix_from_quat(robot_ee_quat_wrt_base)
    rot6d_robot_base_to_robot_ee = rotmat_to_rot6d(rotmat_robot_base_to_robot_ee)

    # End-effector -> Object
    pos_robot_ee_to_obj, quat_robot_ee_to_obj = subtract_frame_transforms(
        t01=robot_ee_pos,
        q01=robot_ee_quat,
        t02=obj_pos,
        q02=obj_quat,
    )
    rotmat_robot_ee_to_obj = matrix_from_quat(quat_robot_ee_to_obj)
    rot6d_robot_ee_to_obj = rotmat_to_rot6d(rotmat_robot_ee_to_obj)

    # Peg -> Peg ends
    _pos_peg_end0, _ = combine_frame_transforms(
        t01=obj_pos,
        q01=obj_quat,
        t12=peg_offset_pos_ends[:, 0],
    )
    _pos_peg_end1, _ = combine_frame_transforms(
        t01=obj_pos,
        q01=obj_quat,
        t12=peg_offset_pos_ends[:, 1],
    )
    pos_peg_ends = torch.stack([_pos_peg_end0, _pos_peg_end1], dim=1)

    # Hole -> Hole entrance | bottom
    pos_hole_entrance, _ = combine_frame_transforms(
        t01=target_pos,
        q01=target_quat,
        t12=hole_offset_pos_entrance,
    )
    pos_hole_bottom, _ = combine_frame_transforms(
        t01=target_pos,
        q01=target_quat,
        t12=hole_offset_pos_bottom,
    )

    # Peg ends -> Hole entrance
    _pos_peg_end0_to_hole_entrance, hole_quat_wrt_peg = subtract_frame_transforms(
        t01=pos_peg_ends[:, 0],
        q01=obj_quat,
        t02=pos_hole_entrance,
        q02=target_quat,
    )
    _pos_peg_end1_to_hole_entrance, _ = subtract_frame_transforms(
        t01=pos_peg_ends[:, 1],
        q01=obj_quat,
        t02=pos_hole_entrance,
    )
    pos_peg_ends_to_hole_entrance = torch.stack(
        [_pos_peg_end0_to_hole_entrance, _pos_peg_end1_to_hole_entrance], dim=1
    )
    rotmat_peg_to_hole = matrix_from_quat(hole_quat_wrt_peg)
    rot6d_peg_to_hole = rotmat_to_rot6d(rotmat_peg_to_hole)

    # Peg ends -> Hole bottom
    _pos_peg_end0_to_hole_bottom, _ = subtract_frame_transforms(
        t01=pos_peg_ends[:, 0],
        q01=obj_quat,
        t02=pos_hole_bottom,
    )
    _pos_peg_end1_to_hole_bottom, _ = subtract_frame_transforms(
        t01=pos_peg_ends[:, 1],
        q01=obj_quat,
        t02=pos_hole_bottom,
    )
    pos_peg_ends_to_hole_bottom = torch.stack(
        [_pos_peg_end0_to_hole_bottom, _pos_peg_end1_to_hole_bottom], dim=1
    )

    # Robot hand wrench
    wrench_robot_hand = robot_incoming_forces[:, robot_joint_indices_hand]
    wrench_robot_hand_mean = (
        wrench_robot_hand.mean(dim=1)
        if torch.numel(wrench_robot_hand)
        else torch.zeros(
            (wrench_robot_hand.shape[0], 1),
            dtype=wrench_robot_hand.dtype,
            device=wrench_robot_hand.device,
        )
    )

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
    )

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
    WEIGHT_OBJ_LIFTED = 8.0
    HEIGHT_OFFSET_OBJ_LIFTED = 0.3
    HEIGHT_SPAN_OBJ_LIFTED = 0.25
    TAHN_STD_HEIGHT_OBJ_LIFTED = 0.05
    obj_target_height_offset = (
        torch.abs(obj_pos[:, 2] - obj_initial_pos_z - HEIGHT_OFFSET_OBJ_LIFTED)
        - HEIGHT_SPAN_OBJ_LIFTED
    ).clamp(min=0.0)
    reward_obj_lifted = WEIGHT_OBJ_LIFTED * (
        1.0 - torch.tanh(obj_target_height_offset / TAHN_STD_HEIGHT_OBJ_LIFTED)
    )

    # Reward: Alignment | Peg -> Hole | Primary Z axis
    WEIGHT_ALIGN_PEG_TO_HOLE_PRIMARY = 8.0
    TANH_STD_ALIGN_PEG_TO_HOLE_PRIMARY = 0.5
    _peg_to_hole_primary_axis_similarity = torch.abs(rotmat_peg_to_hole[:, 2, 2])
    reward_align_peg_to_hole_primary = WEIGHT_ALIGN_PEG_TO_HOLE_PRIMARY * (
        1.0
        - torch.tanh(
            (1.0 - _peg_to_hole_primary_axis_similarity)
            / TANH_STD_ALIGN_PEG_TO_HOLE_PRIMARY
        )
    )

    # Reward: Alignment | Peg -> Hole | Secondary XY axes (affected by primary via power)
    WEIGHT_ALIGN_PEG_TO_HOLE_SECONDARY = 4.0
    TANH_STD_ALIGN_PEG_TO_HOLE_SECONDARY = 0.2
    _peg_to_hole_yaw = torch.atan2(
        rotmat_peg_to_hole[:, 0, 1], rotmat_peg_to_hole[:, 0, 0]
    )
    _symmetry_step = 2 * torch.pi / peg_rot_symmetry_n
    _peg_to_hole_yaw_symmetric_directional = _peg_to_hole_yaw % _symmetry_step
    # Note: Lines above might result in NaN/inf when `peg_rot_symmetry_n=0` (infinite circular symmetry)
    #       However, the following `torch.where()` will handle this case
    _peg_to_hole_yaw_symmetric_normalized = torch.where(
        peg_rot_symmetry_n <= 0,
        0.0,
        torch.min(
            _peg_to_hole_yaw_symmetric_directional,
            _symmetry_step - _peg_to_hole_yaw_symmetric_directional,
        )
        / (_symmetry_step / 2.0),
    )
    reward_align_peg_to_hole_secondary = WEIGHT_ALIGN_PEG_TO_HOLE_SECONDARY * (
        1.0
        - torch.tanh(
            _peg_to_hole_yaw_symmetric_normalized.pow(
                _peg_to_hole_primary_axis_similarity
            )
            / TANH_STD_ALIGN_PEG_TO_HOLE_SECONDARY
        )
    )

    # Reward: Distance | Peg -> Hole entrance
    WEIGHT_DISTANCE_PEG_TO_HOLE_ENTRANCE = 16.0
    TANH_STD_DISTANCE_PEG_TO_HOLE_ENTRANCE = 0.05
    reward_distance_peg_to_hole_entrance = WEIGHT_DISTANCE_PEG_TO_HOLE_ENTRANCE * (
        1.0
        - torch.tanh(
            torch.min(torch.norm(pos_peg_ends_to_hole_entrance, dim=-1), dim=1)[0]
            / TANH_STD_DISTANCE_PEG_TO_HOLE_ENTRANCE
        )
    )

    # Reward: Distance | Peg -> Hole bottom
    WEIGHT_DISTANCE_PEG_TO_HOLE_BOTTOM = 128.0
    TANH_STD_DISTANCE_PEG_TO_HOLE_BOTTOM = 0.005
    reward_distance_peg_to_hole_bottom = WEIGHT_DISTANCE_PEG_TO_HOLE_BOTTOM * (
        1.0
        - torch.tanh(
            torch.min(torch.norm(pos_peg_ends_to_hole_bottom, dim=-1), dim=1)[0]
            / TANH_STD_DISTANCE_PEG_TO_HOLE_BOTTOM
        )
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
                "pos_peg_ends_to_hole_bottom": pos_peg_ends_to_hole_bottom,
                "pos_peg_ends_to_hole_entrance": pos_peg_ends_to_hole_entrance,
                "pos_robot_ee_to_obj": pos_robot_ee_to_obj,
                "rot6d_peg_to_hole": rot6d_peg_to_hole,
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
            "reward_align_peg_to_hole_primary": reward_align_peg_to_hole_primary,
            "reward_align_peg_to_hole_secondary": reward_align_peg_to_hole_secondary,
            "reward_distance_ee_to_obj": reward_distance_ee_to_obj,
            "reward_distance_peg_to_hole_bottom": reward_distance_peg_to_hole_bottom,
            "reward_distance_peg_to_hole_entrance": reward_distance_peg_to_hole_entrance,
            "reward_obj_grasped": reward_obj_grasped,
            "reward_obj_lifted": reward_obj_lifted,
        },
        "term": termination,
        "trunc": truncation,
    }
