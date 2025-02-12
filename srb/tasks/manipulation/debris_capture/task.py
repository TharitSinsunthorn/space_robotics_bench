from dataclasses import MISSING
from typing import Dict, List, Sequence, Tuple

import torch

from srb import assets
from srb._typing import IntermediateTaskState
from srb.core.asset import (
    AssetVariant,
    RigidObject,
    RigidObjectCfg,
    SingleArmManipulator,
    StaticVehicle,
    Terrain,
)
from srb.core.domain import Domain
from srb.core.env import (
    SingleArmEnv,
    SingleArmEnvCfg,
    SingleArmEventCfg,
    SingleArmSceneCfg,
)
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.mdp import reset_root_state_uniform
from srb.core.sensor import ContactSensor, ContactSensorCfg
from srb.core.sim import PreviewSurfaceCfg, SphereCfg
from srb.utils.cfg import configclass
from srb.utils.math import (
    matrix_from_quat,
    rotmat_to_rot6d,
    scale_transform,
    subtract_frame_transforms,
)

##############
### Config ###
##############


@configclass
class SceneCfg(SingleArmSceneCfg):
    obj: RigidObjectCfg = MISSING  # type: ignore
    contacts_robot_hand_obj: ContactSensorCfg = ContactSensorCfg(
        prim_path=MISSING,  # type: ignore
        filter_prim_paths_expr=MISSING,  # type: ignore
    )


@configclass
class EventCfg(SingleArmEventCfg):
    randomize_obj_state: EventTermCfg = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("obj"),
            "pose_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (-0.25, 0.25),
                "roll": (-torch.pi, torch.pi),
                "pitch": (-torch.pi, torch.pi),
                "yaw": (-torch.pi, torch.pi),
            },
            "velocity_range": {
                "x": (-0.2 - 0.05, -0.2 + 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-torch.pi, torch.pi),
                "pitch": (-torch.pi, torch.pi),
                "yaw": (-torch.pi, torch.pi),
            },
        },
    )


@configclass
class TaskCfg(SingleArmEnvCfg):
    ## Scenario
    domain: Domain = Domain.ORBIT

    ## Assets
    terrain: Terrain | AssetVariant | None = None
    vehicle: StaticVehicle | AssetVariant = assets.Gateway()

    ## Scene
    scene: SceneCfg = SceneCfg()

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 10.0
    is_finite_horizon: bool = True

    ## Target
    target_pos: Tuple[float, float, float] = (0.5, 0.0, 0.1)
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
        assert isinstance(self.robot, SingleArmManipulator)

        ## Assets -> Scene
        # Object
        self.scene.obj = assets.rigid_object_from_cfg(
            self,
            seed=self.seed,
            num_assets=self.scene.num_envs,
            prim_path="{ENV_REGEX_NS}/debris",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.5)),
            activate_contact_sensors=True,
        )
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

        ## Cache metrics
        self._robot_joint_indices_arm, _ = self._robot.find_joints(
            self.cfg.robot.regex_joints_arm
        )
        self._robot_joint_indices_hand, _ = (
            self._robot.find_joints(self.cfg.robot.regex_joints_hand)
            if self.cfg.robot.regex_joints_hand
            else ([], [])
        )

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
            obj_pos=self._obj.data.root_com_pos_w,
            obj_quat=self._obj.data.root_com_quat_w,
            obj_vel=self._obj.data.root_com_vel_w,
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
    obj_vel: torch.Tensor,
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

    # Penalty: Object velocity (linear)
    WEIGHT_OBJ_VEL_LINEAR = -2.0
    penalty_obj_vel_linear = WEIGHT_OBJ_VEL_LINEAR * torch.norm(obj_vel[:, :3], dim=-1)

    # Penalty: Object velocity (angular)
    WEIGHT_OBJ_VEL_ANGULAR = -1.0 / (2.0 * torch.pi)
    penalty_obj_vel_angular = WEIGHT_OBJ_VEL_ANGULAR * torch.norm(
        obj_vel[:, 3:], dim=-1
    )

    # Reward: Distance | Object <--> Target
    WEIGHT_DISTANCE_OBJ_TO_TARGET = 32.0
    TANH_STD_DISTANCE_OBJ_TO_TARGET = 0.333
    reward_distance_obj_to_target = WEIGHT_DISTANCE_OBJ_TO_TARGET * (
        1.0
        - torch.tanh(
            torch.norm(pos_obj_to_target, dim=-1) / TANH_STD_DISTANCE_OBJ_TO_TARGET
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
                "pos_obj_to_target": pos_obj_to_target,
                "pos_robot_ee_to_obj": pos_robot_ee_to_obj,
                "rot6d_obj_to_target": rot6d_obj_to_target,
                "rot6d_robot_ee_to_obj": rot6d_robot_ee_to_obj,
                "vel_obj": obj_vel,
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
            "penalty_obj_vel_angular": penalty_obj_vel_angular,
            "penalty_obj_vel_linear": penalty_obj_vel_linear,
            "penalty_undersired_robot_arm_contacts": penalty_undersired_robot_arm_contacts,
            "reward_distance_ee_to_obj": reward_distance_ee_to_obj,
            "reward_distance_obj_to_target": reward_distance_obj_to_target,
            "reward_obj_grasped": reward_obj_grasped,
        },
        "term": termination,
        "trunc": truncation,
    }
