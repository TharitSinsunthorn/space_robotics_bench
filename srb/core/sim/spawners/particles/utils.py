import numpy as np
import torch
from pxr import Usd, UsdGeom, Vt


def particle_positions(prim: Usd.Prim) -> np.ndarray:
    """Get particle positions from USD prim.

    Args:
        prim: USD prim with particles

    Returns:
        Numpy array of particle positions
    """
    return np.array(UsdGeom.Points(prim).GetPointsAttr().Get())


def particle_velocities(prim: Usd.Prim) -> np.ndarray:
    """Get particle velocities from USD prim.

    Args:
        prim: USD prim with particles

    Returns:
        Numpy array of particle velocities
    """
    return np.array(UsdGeom.Points(prim).GetVelocitiesAttr().Get())


def set_particle_positions(prim: Usd.Prim, positions: torch.Tensor | np.ndarray):
    """Set particle positions.

    Args:
        prim: USD prim with particles
        positions: New particle positions
    """
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    UsdGeom.Points(prim).GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))


def set_particle_velocities(prim: Usd.Prim, velocities: torch.Tensor | np.ndarray):
    """Set particle velocities.

    Args:
        prim: USD prim with particles
        velocities: New particle velocities
    """
    if isinstance(velocities, torch.Tensor):
        velocities = velocities.cpu().numpy()
    UsdGeom.Points(prim).GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(velocities))


def count_particles_in_region(
    positions: torch.Tensor | np.ndarray,
    min_bounds: tuple[float, float, float],
    max_bounds: tuple[float, float, float],
) -> int:
    """Count particles within a 3D bounding box region.

    Args:
        positions: Particle positions as (N, 3) array
        min_bounds: Minimum (x, y, z) bounds of region
        max_bounds: Maximum (x, y, z) bounds of region

    Returns:
        Number of particles within the region
    """
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()

    in_x = (positions[:, 0] >= min_bounds[0]) & (positions[:, 0] <= max_bounds[0])
    in_y = (positions[:, 1] >= min_bounds[1]) & (positions[:, 1] <= max_bounds[1])
    in_z = (positions[:, 2] >= min_bounds[2]) & (positions[:, 2] <= max_bounds[2])

    return np.sum(in_x & in_y & in_z)


def calculate_excavated_volume(
    original_positions: torch.Tensor | np.ndarray,
    current_positions: torch.Tensor | np.ndarray,
    excavation_region: tuple[tuple[float, float, float], tuple[float, float, float]],
    particle_volume: float,
) -> float:
    """Calculate volume of material excavated from a region.

    Args:
        original_positions: Original particle positions as (N, 3) array
        current_positions: Current particle positions as (N, 3) array
        excavation_region: Tuple of ((min_x, min_y, min_z), (max_x, max_y, max_z))
        particle_volume: Volume represented by each particle

    Returns:
        Volume of material excavated
    """
    if isinstance(original_positions, torch.Tensor):
        original_positions = original_positions.cpu().numpy()
    if isinstance(current_positions, torch.Tensor):
        current_positions = current_positions.cpu().numpy()

    min_bounds, max_bounds = excavation_region

    original_count = count_particles_in_region(
        original_positions, min_bounds, max_bounds
    )
    current_count = count_particles_in_region(current_positions, min_bounds, max_bounds)

    return max(0, (original_count - current_count) * particle_volume)


def calculate_collection_efficiency(
    original_positions: torch.Tensor | np.ndarray,
    current_positions: torch.Tensor | np.ndarray,
    excavation_region: tuple[tuple[float, float, float], tuple[float, float, float]],
    collection_region: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    """Calculate collection efficiency as ratio of collected to excavated particles.

    Args:
        original_positions: Original particle positions as (N, 3) array
        current_positions: Current particle positions as (N, 3) array
        excavation_region: Tuple of ((min_x, min_y, min_z), (max_x, max_y, max_z))
        collection_region: Tuple of ((min_x, min_y, min_z), (max_x, max_y, max_z))

    Returns:
        Collection efficiency as a value between 0.0 and 1.0
    """
    if isinstance(original_positions, torch.Tensor):
        original_positions = original_positions.cpu().numpy()
    if isinstance(current_positions, torch.Tensor):
        current_positions = current_positions.cpu().numpy()

    excavation_min, excavation_max = excavation_region
    collection_min, collection_max = collection_region

    # Count particles originally in excavation region
    original_in_excavation = count_particles_in_region(
        original_positions, excavation_min, excavation_max
    )

    # Count particles currently in excavation region
    current_in_excavation = count_particles_in_region(
        current_positions, excavation_min, excavation_max
    )

    # Count particles currently in collection region
    current_in_collection = count_particles_in_region(
        current_positions, collection_min, collection_max
    )

    # Calculate particles that were excavated
    excavated_particles = max(0, original_in_excavation - current_in_excavation)

    # Calculate efficiency
    if excavated_particles > 0:
        return current_in_collection / excavated_particles
    else:
        return 0.0


def calculate_particles_displaced(
    original_positions: torch.Tensor | np.ndarray,
    current_positions: torch.Tensor | np.ndarray,
    region: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> int:
    """Calculate how many particles were moved from a region.

    Args:
        original_positions: Original particle positions
        current_positions: Current particle positions
        region: Region to check for displacement

    Returns:
        Number of particles displaced from the region
    """
    if isinstance(original_positions, torch.Tensor):
        original_positions = original_positions.cpu().numpy()
    if isinstance(current_positions, torch.Tensor):
        current_positions = current_positions.cpu().numpy()

    min_bounds, max_bounds = region

    # Find particles that were originally in the region
    in_x = (original_positions[:, 0] >= min_bounds[0]) & (
        original_positions[:, 0] <= max_bounds[0]
    )
    in_y = (original_positions[:, 1] >= min_bounds[1]) & (
        original_positions[:, 1] <= max_bounds[1]
    )
    in_z = (original_positions[:, 2] >= min_bounds[2]) & (
        original_positions[:, 2] <= max_bounds[2]
    )
    in_region_indices = np.where(in_x & in_y & in_z)[0]

    # Check which of these particles are no longer in the region
    if len(in_region_indices) == 0:
        return 0

    particles_to_check = current_positions[in_region_indices]
    in_x = (particles_to_check[:, 0] >= min_bounds[0]) & (
        particles_to_check[:, 0] <= max_bounds[0]
    )
    in_y = (particles_to_check[:, 1] >= min_bounds[1]) & (
        particles_to_check[:, 1] <= max_bounds[1]
    )
    in_z = (particles_to_check[:, 2] >= min_bounds[2]) & (
        particles_to_check[:, 2] <= max_bounds[2]
    )
    still_in_region = in_x & in_y & in_z

    return len(in_region_indices) - np.sum(still_in_region)


def calculate_particle_distance_metrics(
    original_positions: torch.Tensor | np.ndarray,
    current_positions: torch.Tensor | np.ndarray,
    region: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float]:
    """Calculate average and maximum displacement distance for particles originally in region.

    Args:
        original_positions: Original particle positions
        current_positions: Current particle positions
        region: Region to analyze

    Returns:
        Tuple of (average_displacement, max_displacement)
    """
    if isinstance(original_positions, torch.Tensor):
        original_positions = original_positions.cpu().numpy()
    if isinstance(current_positions, torch.Tensor):
        current_positions = current_positions.cpu().numpy()

    min_bounds, max_bounds = region

    # Find particles that were originally in the region
    in_x = (original_positions[:, 0] >= min_bounds[0]) & (
        original_positions[:, 0] <= max_bounds[0]
    )
    in_y = (original_positions[:, 1] >= min_bounds[1]) & (
        original_positions[:, 1] <= max_bounds[1]
    )
    in_z = (original_positions[:, 2] >= min_bounds[2]) & (
        original_positions[:, 2] <= max_bounds[2]
    )
    in_region_indices = np.where(in_x & in_y & in_z)[0]

    if len(in_region_indices) == 0:
        return 0.0, 0.0

    # Calculate displacement distances
    original_pos = original_positions[in_region_indices]
    current_pos = current_positions[in_region_indices]
    displacements = np.linalg.norm(current_pos - original_pos, axis=1)

    return float(np.mean(displacements)), float(np.max(displacements))
