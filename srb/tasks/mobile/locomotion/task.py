from typing import Dict, List, Sequence

import torch

from srb._typing import IntermediateTaskState
from srb.core.asset import LeggedRobot
from srb.core.env import (
    LocomotionEnv,
    LocomotionEnvCfg,
    LocomotionEventCfg,
    LocomotionSceneCfg,
)
from srb.core.manager import EventTermCfg
from srb.core.marker import ARROW_CFG, VisualizationMarkers
from srb.core.mdp import randomize_command
from srb.core.sim import PreviewSurfaceCfg
from srb.utils.cfg import configclass
from srb.utils.math import matrix_from_quat, rotmat_to_rot6d, scale_transform

##############
### Config ###
##############


@configclass
class SceneCfg(LocomotionSceneCfg):
    pass


@configclass
class EventCfg(LocomotionEventCfg):
    command = EventTermCfg(
        func=randomize_command,
        mode="interval",
        is_global_time=True,
        interval_range_s=(0.5, 5.0),
        params={
            "env_attr_name": "_command",
            # "magnitude": 1.0,
        },
    )


@configclass
class TaskCfg(LocomotionEnvCfg):
    ## Scene
    scene: SceneCfg = SceneCfg()

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 20.0
    is_finite_horizon: bool = False

    ## Visualization
    visualize_commands: bool = True


############
### Task ###
############


class Task(LocomotionEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        assert isinstance(self.cfg.robot, LeggedRobot)

        ## Initialize buffers
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

        ## Cache metrics
        self._robot_feet_indices, _ = self._robot.find_bodies(
            self.cfg.robot.regex_links_feet
        )
        self._robot_undesired_contact_body_indices, _ = self._robot.find_bodies(
            self.cfg.robot.regex_links_undesired_contacts
        )

        ## Visualization
        if self.cfg.visualize_commands:
            self._setup_visualization_markers()

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

    def _update_internal_state(self):
        self._internal_state = _compute_internal_state(
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            episode_length_buf=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            truncate_episodes=self.cfg.truncate_episodes,
            robot_joint_pos=self._robot.data.joint_pos,
            robot_joint_acc=self._robot.data.joint_acc,
            robot_joint_applied_torque=self._robot.data.applied_torque,
            robot_quat=self._robot.data.root_quat_w,
            robot_lin_vel=self._robot.data.root_lin_vel_b,
            robot_ang_vel=self._robot.data.root_ang_vel_b,
            robot_projected_gravity=self._robot.data.projected_gravity_b,
            robot_feet_indices=self._robot_feet_indices,
            robot_undesired_contact_body_indices=self._robot_undesired_contact_body_indices,
            contact_net_forces=self._contacts_robot.data.net_forces_w,  # type: ignore
            first_contact=self._contacts_robot.compute_first_contact(self.step_dt),
            last_air_time=self._contacts_robot.data.last_air_time,  # type: ignore
            robot_soft_joint_pos_limits=self._robot.data.soft_joint_pos_limits,
            command=self._command,
        )

        ## Visualization
        if self.cfg.visualize_commands:
            self._update_visualization_markers()

    def _setup_visualization_markers(self):
        ## Linear velocity
        cfg = ARROW_CFG.copy().replace(  # type: ignore
            prim_path="/Visuals/command/target_linvel"
        )
        cfg.markers["arrow"].tail_radius = 0.01
        cfg.markers["arrow"].tail_length = 0.5
        cfg.markers["arrow"].head_radius = 0.02
        cfg.markers["arrow"].head_length = 0.1
        cfg.markers["arrow"].visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.8, 0.0)
        )
        self._marker_target_linvel = VisualizationMarkers(cfg)
        cfg = ARROW_CFG.copy().replace(  # type: ignore
            prim_path="/Visuals/command/robot_linvel"
        )
        cfg.markers["arrow"].tail_radius = 0.01
        cfg.markers["arrow"].tail_length = 0.5
        cfg.markers["arrow"].head_radius = 0.02
        cfg.markers["arrow"].head_length = 0.1
        cfg.markers["arrow"].visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.3, 0.7, 0.3)
        )
        self._marker_robot_linvel = VisualizationMarkers(cfg)

        ## Angular velocity
        cfg = ARROW_CFG.copy().replace(  # type: ignore
            prim_path="/Visuals/command/target_angvel"
        )
        cfg.markers["arrow"].tail_length = 0.0
        cfg.markers["arrow"].tail_radius = 0.0
        cfg.markers["arrow"].head_radius = 0.025
        cfg.markers["arrow"].head_length = 0.15
        cfg.markers["arrow"].visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 0.8)
        )
        self._marker_target_angvel = VisualizationMarkers(cfg)
        cfg = ARROW_CFG.copy().replace(  # type: ignore
            prim_path="/Visuals/command/robot_angvel"
        )
        cfg.markers["arrow"].tail_length = 0.0
        cfg.markers["arrow"].tail_radius = 0.0
        cfg.markers["arrow"].head_radius = 0.025
        cfg.markers["arrow"].head_length = 0.15
        cfg.markers["arrow"].visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.3, 0.3, 0.7)
        )
        self._marker_robot_angvel = VisualizationMarkers(cfg)

    def _update_visualization_markers(self):
        MARKER_OFFSET_Z_LINVEL = 0.2
        MARKER_OFFSET_Z_ANGVEL = 0.175

        ## Common
        robot_pos_w = self._robot.data.root_link_pos_w
        marker_pos = torch.zeros(
            (self.cfg.scene.num_envs, 3), dtype=torch.float32, device=self.device
        )
        marker_orientation = torch.zeros(
            (self.cfg.scene.num_envs, 4), dtype=torch.float32, device=self.device
        )
        marker_scale = torch.ones(
            (self.cfg.scene.num_envs, 3), dtype=torch.float32, device=self.device
        )
        marker_pos[:, :2] = robot_pos_w[:, :2]

        ## Target linear velocity
        marker_pos[:, 2] = robot_pos_w[:, 2] + MARKER_OFFSET_Z_LINVEL
        marker_heading = self._robot.data.heading_w + torch.atan2(
            self._command[:, 1], self._command[:, 0]
        )
        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = torch.norm(
            torch.stack(
                (self._command[:, 0], self._command[:, 1]),
                dim=-1,
            ),
            dim=-1,
        )
        self._marker_target_linvel.visualize(
            marker_pos, marker_orientation, marker_scale
        )

        ## Robot linear velocity
        marker_heading = self._robot.data.heading_w + torch.atan2(
            self._robot.data.root_lin_vel_b[:, 1],
            self._robot.data.root_lin_vel_b[:, 0],
        )
        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=-1)
        self._marker_robot_linvel.visualize(
            marker_pos, marker_orientation, marker_scale
        )

        ## Target angular velocity
        marker_pos[:, 2] = robot_pos_w[:, 2] + MARKER_OFFSET_Z_ANGVEL
        marker_heading = self._robot.data.heading_w + self._command[:, 2]
        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = 1.0
        self._marker_target_angvel.visualize(
            marker_pos, marker_orientation, marker_scale
        )

        ## Robot angular velocity
        marker_heading = (
            self._robot.data.heading_w + self._robot.data.root_ang_vel_w[:, -1]
        )
        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = 1.0
        self._marker_robot_angvel.visualize(
            marker_pos, marker_orientation, marker_scale
        )


@torch.jit.script
def _compute_internal_state(
    *,
    act_current: torch.Tensor,
    act_previous: torch.Tensor,
    command: torch.Tensor,
    contact_net_forces: torch.Tensor,
    episode_length_buf: torch.Tensor,
    first_contact: torch.Tensor,
    robot_joint_pos: torch.Tensor,
    robot_joint_acc: torch.Tensor,
    robot_joint_applied_torque: torch.Tensor,
    last_air_time: torch.Tensor,
    max_episode_length: int,
    robot_feet_indices: List[int],
    robot_lin_vel: torch.Tensor,
    robot_quat: torch.Tensor,
    robot_undesired_contact_body_indices: List[int],
    robot_ang_vel: torch.Tensor,
    robot_projected_gravity: torch.Tensor,
    robot_soft_joint_pos_limits: torch.Tensor,
    truncate_episodes: bool,
) -> (
    IntermediateTaskState
    | Dict[
        str, torch.Tensor | Dict[str, torch.Tensor] | Dict[str, Dict[str, torch.Tensor]]
    ]
):
    # Robot joints
    joint_pos_normalized = scale_transform(
        robot_joint_pos,
        robot_soft_joint_pos_limits[:, :, 0],
        robot_soft_joint_pos_limits[:, :, 1],
    )

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

    # Penalty: Undesired robot contacts
    WEIGHT_UNDERSIRED_CONTACTS = -1.0
    THRESHOLD_UNDERSIRED_CONTACTS = 1.0
    penalty_undersired_contacts = WEIGHT_UNDERSIRED_CONTACTS * (
        torch.max(
            torch.norm(
                contact_net_forces[:, robot_undesired_contact_body_indices, :],
                dim=-1,
            ),
            dim=1,
        )[0]
        > THRESHOLD_UNDERSIRED_CONTACTS
    )

    # Reward: Command tracking (linear)
    WEIGHT_CMD_LIN_VEL_XY = 2.5
    EXP_STD_CMD_LIN_VEL_XY = 0.5
    reward_cmd_lin_vel_xy = WEIGHT_CMD_LIN_VEL_XY * torch.exp(
        -torch.sum(torch.square(command[:, :2] - robot_lin_vel[:, :2]), dim=1)
        / EXP_STD_CMD_LIN_VEL_XY
    )

    # Reward: Command tracking (angular)
    WEIGHT_CMD_ANG_VEL_Z = 1.0
    EXP_STD_CMD_ANG_VEL_Z = 0.25
    reward_cmd_ang_vel_z = WEIGHT_CMD_ANG_VEL_Z * torch.exp(
        -torch.square(command[:, 2] - robot_ang_vel[:, 2]) / EXP_STD_CMD_ANG_VEL_Z
    )

    # Reward: Feet air time
    WEIGHT_FEET_AIR_TIME = 0.5
    THRESHOLD_FEET_AIR_TIME = 0.1
    reward_feet_air_time = (
        WEIGHT_FEET_AIR_TIME
        * (torch.norm(command[:, :2], dim=1) > THRESHOLD_FEET_AIR_TIME)
        * torch.sum(
            (last_air_time[:, robot_feet_indices] - 0.5)
            * first_contact[:, robot_feet_indices],
            dim=1,
        )
    )

    # Penalty: Minimize non-command motion (linear)
    WEIGHT_UNDESIRED_LIN_VEL_Z = -2.0
    penalty_undesired_lin_vel_z = WEIGHT_UNDESIRED_LIN_VEL_Z * torch.square(
        robot_lin_vel[:, 2]
    )

    # Penalty: Minimize non-command motion (angular)
    WEIGHT_UNDESIRED_ANG_VEL_XY = -0.05
    penalty_undesired_ang_vel_xy = WEIGHT_UNDESIRED_ANG_VEL_XY * torch.sum(
        torch.square(robot_ang_vel[:, :2]), dim=-1
    )

    # Penalty: Joint torque
    WEIGHT_JOINT_TORQUE = -0.000025
    penalty_joint_torque = WEIGHT_JOINT_TORQUE * torch.sum(
        torch.square(robot_joint_applied_torque), dim=1
    )

    # Penalty: Joint acceleration
    WEIGHT_JOINT_ACCELERATION = -0.00000025
    penalty_joint_acceleration = WEIGHT_JOINT_ACCELERATION * torch.sum(
        torch.square(robot_joint_acc), dim=1
    )

    # Penalty: Minimize rotation with the gravity direction
    WEIGHT_GRAVITY_ROTATION_ALIGNMENT = -5.0
    penalty_gravity_rotation_alignment = WEIGHT_GRAVITY_ROTATION_ALIGNMENT * torch.sum(
        torch.square(robot_projected_gravity[:, :2]), dim=1
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
            "state_dyn": {
                "contact_net_forces": contact_net_forces,
            },
            "proprio": {
                "rot6d_robot": rot6d_robot,
                "vel_ang_robot": robot_ang_vel,
                "vel_lin_robot": robot_lin_vel,
            },
            "proprio_dyn": {
                "joint_pos": joint_pos_normalized,
            },
            "command": {
                "command": command,
            },
        },
        "rew": {
            "penalty_action_rate": penalty_action_rate,
            "penalty_gravity_rotation_alignment": penalty_gravity_rotation_alignment,
            "penalty_joint_acceleration": penalty_joint_acceleration,
            "penalty_joint_torque": penalty_joint_torque,
            "penalty_undersired_contacts": penalty_undersired_contacts,
            "penalty_undesired_ang_vel_xy": penalty_undesired_ang_vel_xy,
            "penalty_undesired_lin_vel_z": penalty_undesired_lin_vel_z,
            "reward_cmd_ang_vel_z": reward_cmd_ang_vel_z,
            "reward_cmd_lin_vel_xy": reward_cmd_lin_vel_xy,
            "reward_feet_air_time": reward_feet_air_time,
        },
        "term": termination,
        "trunc": truncation,
    }
