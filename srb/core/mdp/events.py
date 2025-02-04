from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import torch
from pxr import Gf

from srb.core.asset import Articulation, RigidObject, RigidObjectCollection, XFormPrim
from srb.core.manager import SceneEntityCfg
from srb.utils.math import quat_from_euler_xyz, quat_mul
from srb.utils.sampling import (
    sample_poisson_disk_2d_looped,
    sample_poisson_disk_3d_looped,
    sample_uniform,
)
from srb.utils.usd import safe_set_attribute_on_usd_prim

if TYPE_CHECKING:
    from srb._typing import AnyEnv


def randomize_command(
    env: "AnyEnv",
    env_ids: torch.Tensor | None,
    env_attr_name: str,
    magnitude: float = 1.0,
):
    if env_ids is None:
        env_ids = torch.arange(
            env.unwrapped.cfg.scene.num_envs,  # type: ignore
            device=env.unwrapped.device,  # type: ignore
        )
    cmd_attr = getattr(env, env_attr_name)
    cmd_attr[env_ids] = sample_uniform(
        -magnitude,
        magnitude,
        cmd_attr.shape,
        device=env.unwrapped.device,  # type: ignore
    )


def reset_xform_orientation_uniform(
    env: "AnyEnv",
    env_ids: torch.Tensor,
    orientation_distribution_params: Dict[str, Tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    asset: XFormPrim = env.scene[asset_cfg.name]

    range_list = [
        orientation_distribution_params.get(key, (0.0, 0.0))
        for key in ["roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset._device)
    rand_samples = sample_uniform(
        ranges[:, 0], ranges[:, 1], (1, 3), device=asset._device
    )

    orientations = quat_from_euler_xyz(
        rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2]
    )

    asset.set_world_poses(orientations=orientations)


def reset_joints_by_offset(
    env: "AnyEnv",
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
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
    joint_pos += sample_uniform(*position_range, joint_pos.shape, joint_pos.device)  # type: ignore
    joint_vel += sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)  # type: ignore

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # Clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # Set into the physics simulation
    joint_indices = asset.find_joints(asset_cfg.joint_names)[0]  # type: ignore
    asset.write_joint_state_to_sim(
        joint_pos[:, joint_indices],
        joint_vel[:, joint_indices],
        joint_ids=joint_indices,
        env_ids=env_ids,  # type: ignore
    )


def randomize_usd_prim_attribute_uniform(
    env: "AnyEnv",
    env_ids: torch.Tensor | None,
    attr_name: str,
    distribution_params: Tuple[float | Sequence[float], float | Sequence[float]],
    asset_cfg: SceneEntityCfg,
):
    asset: XFormPrim = env.scene[asset_cfg.name]
    if isinstance(distribution_params[0], Sequence):
        dist_len = len(distribution_params[0])
        distribution_params = (  # type: ignore
            torch.tensor(distribution_params[0]),
            torch.tensor(distribution_params[1]),
        )
    else:
        dist_len = 1
    for i, prim in enumerate(asset.prims):
        if env_ids and i not in env_ids:
            continue
        value = sample_uniform(
            distribution_params[0],  # type: ignore
            distribution_params[1],  # type: ignore
            (dist_len,),
            device="cpu",
        )
        value = value.item() if dist_len == 1 else value.tolist()
        safe_set_attribute_on_usd_prim(
            prim, f"inputs:{attr_name}", value, camel_case=True
        )


def randomize_gravity_uniform(
    env: "AnyEnv",
    env_ids: torch.Tensor | None,
    distribution_params: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
):
    physics_scene = env.sim._physics_context._physics_scene  # type: ignore
    gravity = sample_uniform(
        torch.tensor(distribution_params[0]),
        torch.tensor(distribution_params[1]),
        (3,),
        device="cpu",
    )
    gravity_magnitude = torch.norm(gravity)
    if gravity_magnitude == 0.0:
        gravity_direction = gravity
    else:
        gravity_direction = gravity / gravity_magnitude

    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(*gravity_direction.tolist()))
    physics_scene.CreateGravityMagnitudeAttr(gravity_magnitude.item())


def follow_xform_orientation_linear_trajectory(
    env: "AnyEnv",
    env_ids: torch.Tensor,
    orientation_step_params: Dict[str, float],
    asset_cfg: SceneEntityCfg,
):
    asset: XFormPrim = env.scene[asset_cfg.name]

    _, current_quat = asset.get_world_poses()

    steps = torch.tensor(
        [orientation_step_params.get(key, 0.0) for key in ["roll", "pitch", "yaw"]],
        device=asset._device,
    )
    step_quat = quat_from_euler_xyz(steps[0], steps[1], steps[2]).unsqueeze(0)

    orientations = quat_mul(current_quat, step_quat)  # type: ignore

    asset.set_world_poses(orientations=orientations)


def reset_root_state_uniform_poisson_disk_2d(
    env: "AnyEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    radius: float,
    asset_cfg: List[SceneEntityCfg],
):
    # Extract the used quantities (to enable type-hinting)
    assets: List[RigidObject | Articulation] = [
        env.scene[cfg.name] for cfg in asset_cfg
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
        sample_poisson_disk_2d_looped(
            (len(env_ids), len(asset_cfg)),
            (
                (range_list[0][0], range_list[1][0]),
                (range_list[0][1], range_list[1][1]),
            ),
            radius,
        ),
        device=assets[0].device,
    )
    rand_samples = sample_uniform(
        ranges[2:, 0],
        ranges[2:, 1],
        (len(env_ids), len(asset_cfg), 4),
        device=assets[0].device,
    )
    rand_samples = torch.cat([samples_pos_xy, rand_samples], dim=-1)

    positions = (
        root_states[:, :, 0:3]
        + env.scene.env_origins[env_ids].repeat(len(asset_cfg), 1, 1).swapaxes(0, 1)
        + rand_samples[:, :, 0:3]
    )
    orientations_delta = quat_from_euler_xyz(
        rand_samples[:, :, 3], rand_samples[:, :, 4], rand_samples[:, :, 5]
    )
    orientations = quat_mul(root_states[:, :, 3:7], orientations_delta)

    # Velocities
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, dtype=torch.float32, device=assets[0].device)
    rand_samples = sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), len(asset_cfg), 6),
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
            torch.cat([position, orientation], dim=-1),
            env_ids=env_ids,  # type: ignore
        )
        asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)  # type: ignore


def reset_collection_root_state_uniform_poisson_disk_2d(
    env: "AnyEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    radius: float,
    asset_cfg: SceneEntityCfg,
):
    # Extract the used quantities (to enable type-hinting)
    assets: RigidObjectCollection = env.scene[asset_cfg.name]

    # Get default root state
    root_states = assets.data.default_object_state[env_ids]

    # Poses
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, dtype=torch.float32, device=assets.device)
    samples_pos_xy = torch.tensor(
        sample_poisson_disk_2d_looped(
            (len(env_ids), assets.num_objects),
            (
                (range_list[0][0], range_list[1][0]),
                (range_list[0][1], range_list[1][1]),
            ),
            radius,
        ),
        device=assets.device,
    )
    rand_samples = sample_uniform(
        ranges[2:, 0],
        ranges[2:, 1],
        (len(env_ids), assets.num_objects, 4),
        device=assets.device,
    )
    rand_samples = torch.cat([samples_pos_xy, rand_samples], dim=-1)

    positions = (
        root_states[:, :, 0:3]
        + env.scene.env_origins[env_ids].repeat(assets.num_objects, 1, 1).swapaxes(0, 1)
        + rand_samples[:, :, 0:3]
    )
    orientations_delta = quat_from_euler_xyz(
        rand_samples[:, :, 3], rand_samples[:, :, 4], rand_samples[:, :, 5]
    )
    orientations = quat_mul(root_states[:, :, 3:7], orientations_delta)

    # Velocities
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, dtype=torch.float32, device=assets.device)
    rand_samples = sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), assets.num_objects, 6),
        device=assets.device,
    )
    velocities = root_states[:, :, 7:13] + rand_samples

    # Set into the physics simulation
    assets.write_object_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    assets.write_object_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_uniform_poisson_disk_3d(
    env: "AnyEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float, float]],
    velocity_range: dict[str, tuple[float, float, float]],
    radius: float,
    asset_cfg: List[SceneEntityCfg],
):
    # Extract the used quantities (to enable type-hinting)
    assets: List[RigidObject | Articulation] = [
        env.scene[cfg.name] for cfg in asset_cfg
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
    samples_pos = torch.tensor(
        sample_poisson_disk_3d_looped(
            (len(env_ids), len(asset_cfg)),
            (
                (range_list[0][0], range_list[1][0], range_list[2][0]),
                (range_list[0][1], range_list[1][1], range_list[2][1]),
            ),
            radius,
        ),
        device=assets[0].device,
    )
    rand_samples = sample_uniform(
        ranges[3:, 0],
        ranges[3:, 1],
        (len(env_ids), len(asset_cfg), 3),
        device=assets[0].device,
    )
    rand_samples = torch.cat([samples_pos, rand_samples], dim=-1)

    positions = (
        root_states[:, :, 0:3]
        + env.scene.env_origins[env_ids].repeat(len(asset_cfg), 1, 1).swapaxes(0, 1)
        + rand_samples[:, :, 0:3]
    )
    orientations_delta = quat_from_euler_xyz(
        rand_samples[:, :, 3], rand_samples[:, :, 4], rand_samples[:, :, 5]
    )
    orientations = quat_mul(root_states[:, :, 3:7], orientations_delta)

    # Velocities
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, dtype=torch.float32, device=assets[0].device)
    rand_samples = sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), len(asset_cfg), 6),
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
            torch.cat([position, orientation], dim=-1),
            env_ids=env_ids,  # type: ignore
        )
        asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)  # type: ignore


def reset_collection_root_state_uniform_poisson_disk_3d(
    env: "AnyEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float, float]],
    velocity_range: dict[str, tuple[float, float, float]],
    radius: float,
    asset_cfg: SceneEntityCfg,
):
    # Extract the used quantities (to enable type-hinting)
    assets: RigidObjectCollection = env.scene[asset_cfg.name]

    # Get default root state
    root_states = assets.data.default_object_state[env_ids]

    # Poses
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, dtype=torch.float32, device=assets.device)
    samples_pos = torch.tensor(
        sample_poisson_disk_3d_looped(
            (len(env_ids), assets.num_objects),
            (
                (range_list[0][0], range_list[1][0], range_list[2][0]),
                (range_list[0][1], range_list[1][1], range_list[2][1]),
            ),
            radius,
        ),
        device=assets.device,
    )
    rand_samples = sample_uniform(
        ranges[3:, 0],
        ranges[3:, 1],
        (len(env_ids), assets.num_objects, 3),
        device=assets.device,
    )
    rand_samples = torch.cat([samples_pos, rand_samples], dim=-1)

    positions = (
        root_states[:, :, 0:3]
        + env.scene.env_origins[env_ids].repeat(assets.num_objects, 1, 1).swapaxes(0, 1)
        + rand_samples[:, :, 0:3]
    )
    orientations_delta = quat_from_euler_xyz(
        rand_samples[:, :, 3], rand_samples[:, :, 4], rand_samples[:, :, 5]
    )
    orientations = quat_mul(root_states[:, :, 3:7], orientations_delta)

    # Velocities
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, dtype=torch.float32, device=assets.device)
    rand_samples = sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), assets.num_objects, 6),
        device=assets.device,
    )
    velocities = root_states[:, :, 7:13] + rand_samples

    # Set into the physics simulation
    assets.write_object_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    assets.write_object_velocity_to_sim(velocities, env_ids=env_ids)
