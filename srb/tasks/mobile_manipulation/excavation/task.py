from dataclasses import MISSING
from typing import Dict, Sequence, Tuple

import torch

from srb import assets
from srb._typing import StepReturn
from srb.core.asset import AssetBaseCfg, AssetVariant, GroundManipulator
from srb.core.env import (
    GroundManipulationEnv,
    GroundManipulationEnvCfg,
    GroundManipulationEventCfg,
    GroundManipulationSceneCfg,
    ViewerCfg,
)
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import CUBOID_CFG, VisualizationMarkers
from srb.core.mdp import push_by_setting_velocity
from srb.core.sim import PyramidParticlesSpawnerCfg, PreviewSurfaceCfg
from srb.core.sim.spawners.particles.utils import (
    particle_positions,
    count_particles_in_region,
    calculate_excavated_volume,
    calculate_collection_efficiency,
)
from srb.utils.cfg import configclass
from srb.utils.math import (
    matrix_from_quat,
    rotmat_to_rot6d,
    rpy_to_quat,
    scale_transform,
)

##############
### Config ###
##############


@configclass
class ExcavationRegionsCfg:
    """Configuration for excavation and collection regions."""

    # Target excavation region (x_min, y_min, z_min), (x_max, y_max, z_max)
    excavation_region: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        (-0.5, -0.5, 0.0),
        (0.5, 0.5, 0.3),
    )
    # Target collection region (x_min, y_min, z_min), (x_max, y_max, z_max)
    collection_region: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        (1.0, -0.5, 0.0),
        (2.0, 0.5, 0.3),
    )
    # Volume represented by each particle
    particle_volume: float = 1e-4
    # Visualization settings
    visualize_regions: bool = True
    excavation_region_color: Tuple[float, float, float] = (0.8, 0.2, 0.2)
    collection_region_color: Tuple[float, float, float] = (0.2, 0.8, 0.2)


@configclass
class SceneCfg(GroundManipulationSceneCfg):
    env_spacing: float = 8.0

    regolith: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/regolith",
        spawn=PyramidParticlesSpawnerCfg(
            ratio=0.5,
            particle_size=0.01,
            dim_x=MISSING,  # type: ignore
            dim_y=MISSING,  # type: ignore
            dim_z=12,
            velocity=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.0)),
            fluid=False,
            friction=1.0,
            cohesion=0.5,
            cast_shadows=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
    )


@configclass
class EventCfg(GroundManipulationEventCfg):
    push_robot: EventTermCfg = EventTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
        },
    )


@configclass
class TaskCfg(GroundManipulationEnvCfg):
    ## Assets
    robot: GroundManipulator | AssetVariant = assets.GenericGroundManipulator(
        mobile_base=assets.Spot(),
        manipulator=assets.Franka(end_effector=assets.Scoop()),
    )
    robot.asset_cfg.init_state.pos = (-1.0, 0.0, 1.0)  # type: ignore
    robot.asset_cfg.init_state.rot = rpy_to_quat(0.0, 0.0, 0.0)  # type: ignore

    ## Scene
    scene: SceneCfg = SceneCfg()

    ## Events
    events: EventCfg = EventCfg()

    ## Excavation regions
    regions: ExcavationRegionsCfg = ExcavationRegionsCfg()

    ## Reward weights
    reward_weights: Dict[str, float] = {
        "excavation": 1.0,  # Reward for excavating material
        "collection": 2.0,  # Reward for collecting excavated material
        "efficiency": 0.5,  # Reward for efficiency (collected/excavated)
        "action_rate": -0.05,  # Penalty for large action changes
        "joint_torque": -0.000025,  # Penalty for high joint torque
        "joint_acceleration": -0.00025,  # Penalty for high joint acceleration
        "undesired_contacts": -2.0,  # Penalty for undesired robot contacts
    }

    ## Time
    episode_length_s: float = 30.0
    is_finite_horizon: bool = True

    viewer: ViewerCfg = ViewerCfg(
        eye=(3.0, 4.0, 1.8), lookat=(0.5, 0.0, 0.3), origin_type="env"
    )

    def __post_init__(self):
        super().__post_init__()

        # Scene: Regolith
        _regolith_dim = int(
            self.spacing / self.scene.regolith.spawn.particle_size  # type: ignore
        )
        self.scene.regolith.spawn.dim_x = _regolith_dim  # type: ignore
        self.scene.regolith.spawn.dim_y = _regolith_dim  # type: ignore


############
### Task ###
############


class Task(GroundManipulationEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Store the initial particle positions
        self._initial_particle_positions = None
        self._prev_excavated_volume = None
        self._prev_collected_volume = None

        # Get regolith prim path for each env
        self._regolith_prims = []
        for i in range(self.num_envs):
            env_ns = self._scene_manager.get_env_namespace(i)
            regolith_path = self.cfg.scene.regolith.prim_path.format(
                ENV_REGEX_NS=env_ns
            )
            self._regolith_prims.append(self._stage.GetPrimAtPath(regolith_path))

        # Set up visualization for excavation and collection regions if enabled
        if self.cfg.regions.visualize_regions:
            self._setup_region_visualization()

    def _setup_region_visualization(self):
        """Set up visualization markers for excavation and collection regions."""
        # Excavation region marker
        excavation_min, excavation_max = self.cfg.regions.excavation_region
        excavation_pos = [
            (excavation_min[0] + excavation_max[0]) / 2,
            (excavation_min[1] + excavation_max[1]) / 2,
            (excavation_min[2] + excavation_max[2]) / 2,
        ]
        excavation_size = [
            excavation_max[0] - excavation_min[0],
            excavation_max[1] - excavation_min[1],
            excavation_max[2] - excavation_min[2],
        ]

        cfg = CUBOID_CFG.copy().replace(  # type: ignore
            prim_path="/Visuals/excavation_region"
        )
        cfg.markers["cuboid"].visual_material = PreviewSurfaceCfg(
            diffuse_color=self.cfg.regions.excavation_region_color,
            metallic=0.0,
            roughness=0.8,
            opacity=0.3,
        )
        self._excavation_marker = VisualizationMarkers(cfg)

        # Collection region marker
        collection_min, collection_max = self.cfg.regions.collection_region
        collection_pos = [
            (collection_min[0] + collection_max[0]) / 2,
            (collection_min[1] + collection_max[1]) / 2,
            (collection_min[2] + collection_max[2]) / 2,
        ]
        collection_size = [
            collection_max[0] - collection_min[0],
            collection_max[1] - collection_min[1],
            collection_max[2] - collection_min[2],
        ]

        cfg = CUBOID_CFG.copy().replace(  # type: ignore
            prim_path="/Visuals/collection_region"
        )
        cfg.markers["cuboid"].visual_material = PreviewSurfaceCfg(
            diffuse_color=self.cfg.regions.collection_region_color,
            metallic=0.0,
            roughness=0.8,
            opacity=0.3,
        )
        self._collection_marker = VisualizationMarkers(cfg)

        # Visualize markers
        excavation_pos_tensor = torch.tensor(
            [excavation_pos], dtype=torch.float32, device=self.device
        )
        excavation_quat_tensor = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device
        )
        excavation_scale_tensor = torch.tensor(
            [excavation_size], dtype=torch.float32, device=self.device
        )

        collection_pos_tensor = torch.tensor(
            [collection_pos], dtype=torch.float32, device=self.device
        )
        collection_quat_tensor = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device
        )
        collection_scale_tensor = torch.tensor(
            [collection_size], dtype=torch.float32, device=self.device
        )

        self._excavation_marker.visualize(
            excavation_pos_tensor, excavation_quat_tensor, excavation_scale_tensor
        )
        self._collection_marker.visualize(
            collection_pos_tensor, collection_quat_tensor, collection_scale_tensor
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        # Record initial particle positions after reset
        if self._initial_particle_positions is None:
            self._initial_particle_positions = [None] * self.num_envs
            self._prev_excavated_volume = torch.zeros(self.num_envs, device=self.device)
            self._prev_collected_volume = torch.zeros(self.num_envs, device=self.device)

        # Store initial positions for the reset environments
        for i in env_ids:
            self._initial_particle_positions[i] = particle_positions(
                self._regolith_prims[i]
            )
            self._prev_excavated_volume[i] = 0.0
            self._prev_collected_volume[i] = 0.0

    def compute_excavation_metrics(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute excavation metrics for all environments.

        Returns:
            Tuple of (excavated_volume, collected_volume, efficiency)
        """
        excavated_volume = torch.zeros(self.num_envs, device=self.device)
        collected_volume = torch.zeros(self.num_envs, device=self.device)
        collection_efficiency = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.num_envs):
            if self._initial_particle_positions[i] is None:
                continue

            # Get current particle positions
            current_positions = particle_positions(self._regolith_prims[i])

            # Calculate particles in excavation and collection regions
            excavation_min, excavation_max = self.cfg.regions.excavation_region
            collection_min, collection_max = self.cfg.regions.collection_region

            original_in_excavation = count_particles_in_region(
                self._initial_particle_positions[i], excavation_min, excavation_max
            )
            current_in_excavation = count_particles_in_region(
                current_positions, excavation_min, excavation_max
            )
            current_in_collection = count_particles_in_region(
                current_positions, collection_min, collection_max
            )

            # Calculate volumes
            excavated = (
                original_in_excavation - current_in_excavation
            ) * self.cfg.regions.particle_volume
            collected = current_in_collection * self.cfg.regions.particle_volume

            # Calculate efficiency (avoid division by zero)
            if excavated > 0:
                efficiency = collected / excavated
            else:
                efficiency = 0.0

            excavated_volume[i] = excavated
            collected_volume[i] = collected
            collection_efficiency[i] = efficiency

        return excavated_volume, collected_volume, collection_efficiency

    def extract_step_return(self) -> StepReturn:
        # Calculate excavation metrics
        excavated_volume, collected_volume, collection_efficiency = (
            self.compute_excavation_metrics()
        )

        # Calculate reward delta (change since last step)
        excavated_delta = excavated_volume - self._prev_excavated_volume
        collected_delta = collected_volume - self._prev_collected_volume

        # Store current values for next step
        self._prev_excavated_volume = excavated_volume
        self._prev_collected_volume = collected_volume

        return _compute_step_return(
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            episode_length=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            joint_pos_robot=self._robot.data.joint_pos,
            joint_pos_limits_robot=(
                self._robot.data.soft_joint_pos_limits
                if torch.all(torch.isfinite(self._robot.data.soft_joint_pos_limits))
                else None
            ),
            joint_acc_robot=self._robot.data.joint_acc,
            joint_applied_torque_robot=self._robot.data.applied_torque,
            tf_pos_robot=self._robot.data.root_pos_w,
            tf_quat_robot=self._robot.data.root_quat_w,
            vel_lin_robot=self._robot.data.root_lin_vel_b,
            vel_ang_robot=self._robot.data.root_ang_vel_b,
            contact_forces_robot=self._contacts_robot.data.net_forces_w,  # type: ignore
            imu_lin_acc=self._imu_robot.data.lin_acc_b,
            imu_ang_vel=self._imu_robot.data.ang_vel_b,
            truncate_episodes=self.cfg.truncate_episodes,
            excavated_volume=excavated_volume,
            excavated_delta=excavated_delta,
            collected_volume=collected_volume,
            collected_delta=collected_delta,
            collection_efficiency=collection_efficiency,
            reward_weights=self.cfg.reward_weights,
        )


@torch.jit.script
def _compute_step_return(
    *,
    # Actions
    act_current: torch.Tensor,
    act_previous: torch.Tensor,
    # Time
    episode_length: torch.Tensor,
    max_episode_length: int,
    # Robot state
    joint_pos_robot: torch.Tensor,
    joint_pos_limits_robot: torch.Tensor | None,
    joint_acc_robot: torch.Tensor,
    joint_applied_torque_robot: torch.Tensor,
    tf_pos_robot: torch.Tensor,
    tf_quat_robot: torch.Tensor,
    vel_lin_robot: torch.Tensor,
    vel_ang_robot: torch.Tensor,
    contact_forces_robot: torch.Tensor,
    imu_lin_acc: torch.Tensor,
    imu_ang_vel: torch.Tensor,
    # Task state
    excavated_volume: torch.Tensor,
    excavated_delta: torch.Tensor,
    collected_volume: torch.Tensor,
    collected_delta: torch.Tensor,
    collection_efficiency: torch.Tensor,
    # Config
    truncate_episodes: bool,
    reward_weights: Dict[str, float],
) -> StepReturn:
    # Transform joint positions to normalized space
    if joint_pos_limits_robot is not None:
        joint_pos_normalized = scale_transform(
            joint_pos_robot,
            joint_pos_limits_robot[:, :, 0],
            joint_pos_limits_robot[:, :, 1],
        )
    else:
        joint_pos_normalized = joint_pos_robot

    # Robot pose
    rotmat_robot = matrix_from_quat(tf_quat_robot)
    rot6d_robot = rotmat_to_rot6d(rotmat_robot)

    # Contact forces
    contact_forces_mean_robot = contact_forces_robot.mean(dim=1)

    #############
    ## Rewards ##
    #############
    # Reward: Excavation (reward for newly excavated volume)
    reward_excavation = reward_weights["excavation"] * excavated_delta

    # Reward: Collection (reward for newly collected volume)
    reward_collection = reward_weights["collection"] * collected_delta

    # Reward: Efficiency (reward for collection efficiency)
    reward_efficiency = reward_weights["efficiency"] * collection_efficiency

    # Penalty: Action rate
    penalty_action_rate = reward_weights["action_rate"] * torch.sum(
        torch.square(act_current - act_previous), dim=1
    )

    # Penalty: Joint torque
    WEIGHT_JOINT_TORQUE = reward_weights["joint_torque"]
    MAX_JOINT_TORQUE_PENALTY = -4.0
    penalty_joint_torque = torch.clamp_min(
        WEIGHT_JOINT_TORQUE
        * torch.sum(torch.square(joint_applied_torque_robot), dim=1),
        min=MAX_JOINT_TORQUE_PENALTY,
    )

    # Penalty: Joint acceleration
    WEIGHT_JOINT_ACCELERATION = reward_weights["joint_acceleration"]
    MAX_JOINT_ACCELERATION_PENALTY = -2.0
    penalty_joint_acceleration = torch.clamp_min(
        WEIGHT_JOINT_ACCELERATION * torch.sum(torch.square(joint_acc_robot), dim=1),
        min=MAX_JOINT_ACCELERATION_PENALTY,
    )

    # Penalty: Undesired robot contacts
    WEIGHT_UNDESIRED_ROBOT_CONTACTS = reward_weights["undesired_contacts"]
    THRESHOLD_UNDESIRED_ROBOT_CONTACTS = 5.0
    penalty_undesired_robot_contacts = WEIGHT_UNDESIRED_ROBOT_CONTACTS * (
        torch.max(torch.norm(contact_forces_robot, dim=-1), dim=1)[0]
        > THRESHOLD_UNDESIRED_ROBOT_CONTACTS
    )

    # Total reward
    reward = (
        reward_excavation
        + reward_collection
        + reward_efficiency
        + penalty_action_rate
        + penalty_joint_torque
        + penalty_joint_acceleration
        + penalty_undesired_robot_contacts
    )

    ##################
    ## Terminations ##
    ##################
    termination = torch.zeros(
        episode_length.size(0), dtype=torch.bool, device=episode_length.device
    )
    truncation = (
        episode_length >= max_episode_length
        if truncate_episodes
        else torch.zeros_like(termination)
    )

    return StepReturn(
        {
            # State observations
            "state": {
                "rot6d_robot": rot6d_robot,
                "vel_lin_robot": vel_lin_robot,
                "vel_ang_robot": vel_ang_robot,
                "contact_forces_mean_robot": contact_forces_mean_robot,
            },
            "state_dyn": {
                "contact_forces_robot": contact_forces_robot,
            },
            # Task-specific observations
            "excavation": {
                "excavated_volume": excavated_volume,
                "collected_volume": collected_volume,
                "collection_efficiency": collection_efficiency,
            },
            # Proprioceptive observations
            "proprio": {
                "imu_lin_acc": imu_lin_acc,
                "imu_ang_vel": imu_ang_vel,
            },
            "proprio_dyn": {
                "joint_pos": joint_pos_normalized,
                "joint_acc": joint_acc_robot,
                "joint_applied_torque": joint_applied_torque_robot,
            },
        },
        {
            "reward_excavation": reward_excavation,
            "reward_collection": reward_collection,
            "reward_efficiency": reward_efficiency,
            "penalty_action_rate": penalty_action_rate,
            "penalty_joint_torque": penalty_joint_torque,
            "penalty_joint_acceleration": penalty_joint_acceleration,
            "penalty_undesired_robot_contacts": penalty_undesired_robot_contacts,
            "reward": reward,
        },
        termination,
        truncation,
    )
