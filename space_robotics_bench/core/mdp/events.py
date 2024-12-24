from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from omni.isaac.core.prims.xform_prim_view import XFormPrimView
from omni.isaac.lab.managers import SceneEntityCfg

import space_robotics_bench.utils.math as math_utils
import space_robotics_bench.utils.sampling as sampling_utils
from space_robotics_bench.core.asset import Articulation, RigidObject

if TYPE_CHECKING:
    from space_robotics_bench.core.envs import BaseEnv


def reset_xform_orientation_uniform(
    env: "BaseEnv",
    env_ids: torch.Tensor,
    orientation_distribution_params: Dict[str, Tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    asset: XFormPrimView = env.scene[asset_cfg.name]

    range_list = [
        orientation_distribution_params.get(key, (0.0, 0.0))
        for key in ["roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset._device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (1, 3), device=asset._device
    )

    orientations = math_utils.quat_from_euler_xyz(
        rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2]
    )

    asset.set_world_poses(orientations=orientations)


def follow_xform_orientation_linear_trajectory(
    env: "BaseEnv",
    env_ids: torch.Tensor,
    orientation_step_params: Dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    asset: XFormPrimView = env.scene[asset_cfg.name]

    _, current_quat = asset.get_world_poses()

    steps = torch.tensor(
        [orientation_step_params.get(key, 0.0) for key in ["roll", "pitch", "yaw"]],
        device=asset._device,
    )
    step_quat = math_utils.quat_from_euler_xyz(steps[0], steps[1], steps[2]).unsqueeze(
        0
    )

    orientations = math_utils.quat_mul(current_quat, step_quat)

    asset.set_world_poses(orientations=orientations)


def reset_joints_by_offset(
    env: "BaseEnv",
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # Bias these values randomly
    joint_pos += math_utils.sample_uniform(
        *position_range, joint_pos.shape, joint_pos.device
    )
    joint_vel += math_utils.sample_uniform(
        *velocity_range, joint_vel.shape, joint_vel.device
    )

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # Clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # Set into the physics simulation
    joint_indices = asset.find_joints(asset_cfg.joint_names)[0]
    asset.write_joint_state_to_sim(
        joint_pos[:, joint_indices],
        joint_vel[:, joint_indices],
        joint_ids=joint_indices,
        env_ids=env_ids,
    )


def reset_root_state_uniform_poisson_disk_2d(
    env: "BaseEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    radius: float,
    asset_cfgs: List[SceneEntityCfg] = [
        SceneEntityCfg("robot"),
    ],
):
    # Extract the used quantities (to enable type-hinting)
    assets: List[RigidObject | Articulation] = [
        env.scene[asset_cfg.name] for asset_cfg in asset_cfgs
    ]
    # Get default root state
    root_states = torch.stack(
        [asset.data.default_root_state[env_ids].clone() for asset in assets],
    ).swapaxes(0, 1)

    # Poses
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, dtype=torch.float32, device=assets[0].device)
    samples_pos_xy = torch.tensor(
        sampling_utils.sample_poisson_disk_2d_looped(
            (len(env_ids), len(asset_cfgs)),
            (
                (range_list[0][0], range_list[1][0]),
                (range_list[0][1], range_list[1][1]),
            ),
            radius,
        ),
        device=assets[0].device,
    )
    rand_samples = math_utils.sample_uniform(
        ranges[2:, 0],
        ranges[2:, 1],
        (len(env_ids), len(asset_cfgs), 4),
        device=assets[0].device,
    )
    rand_samples = torch.cat([samples_pos_xy, rand_samples], dim=-1)

    positions = (
        root_states[:, :, 0:3]
        + env.scene.env_origins[env_ids].repeat(len(asset_cfgs), 1, 1).swapaxes(0, 1)
        + rand_samples[:, :, 0:3]
    )
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, :, 3], rand_samples[:, :, 4], rand_samples[:, :, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, :, 3:7], orientations_delta)

    # Velocities
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, dtype=torch.float32, device=assets[0].device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), len(asset_cfgs), 6),
        device=assets[0].device,
    )
    velocities = root_states[:, :, 7:13] + rand_samples

    # Set into the physics simulation
    for asset, position, orientation, velocity in zip(
        assets,
        positions.unbind(1),
        orientations.unbind(1),
        velocities.unbind(1),
    ):
        asset.write_root_pose_to_sim(
            torch.cat([position, orientation], dim=-1), env_ids=env_ids
        )
        asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)
