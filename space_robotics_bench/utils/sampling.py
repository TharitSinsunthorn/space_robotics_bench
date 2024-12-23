from typing import Iterable, List, Tuple

import numpy as np
import torch
from oxidasim.sampling import *  # noqa: F403
from pxr import Gf


def compute_grid_spacing(
    num_instances: int,
    spacing: float,
    global_pos_offset: np.ndarray | torch.Tensor | Iterable | None = None,
    global_rot_offset: np.ndarray | torch.Tensor | Iterable | None = None,
) -> Tuple[Tuple[int, int], Tuple[List[float], List[float]]]:
    if global_pos_offset is not None:
        if isinstance(global_pos_offset, torch.Tensor):
            global_pos_offset = global_pos_offset.detach().cpu().numpy()
        elif not isinstance(global_pos_offset, np.ndarray):
            global_pos_offset = np.asarray(global_pos_offset)
    if global_rot_offset is not None:
        if isinstance(global_rot_offset, torch.Tensor):
            global_rot_offset = global_rot_offset.detach().cpu().numpy()
        elif not isinstance(global_rot_offset, np.ndarray):
            global_rot_offset = np.asarray(global_rot_offset)

    num_per_row = np.ceil(np.sqrt(num_instances))
    num_rows = np.ceil(num_instances / num_per_row)
    num_cols = np.ceil(num_instances / num_rows)

    row_offset = 0.5 * spacing * (num_rows - 1)
    col_offset = 0.5 * spacing * (num_cols - 1)

    positions = []
    orientations = []

    for i in range(num_instances):
        row = i // num_cols
        col = i % num_cols
        x = row_offset - row * spacing
        y = col * spacing - col_offset

        position = [x, y, 0]
        if global_pos_offset is not None:
            translation = global_pos_offset + position
        else:
            translation = position
        positions.append(translation)

        orientation: Gf.Quatd = Gf.Quatd.GetIdentity()  # type: ignore
        if global_rot_offset is not None:
            orientation = (
                Gf.Quatd(
                    global_rot_offset[0].item(),
                    Gf.Vec3d(global_rot_offset[1:].tolist()),
                )
                * orientation
            )
        orientations.append(
            [
                orientation.GetReal(),
                orientation.GetImaginary()[0],
                orientation.GetImaginary()[1],
                orientation.GetImaginary()[2],
            ]
        )

    return ((num_rows, num_cols), (positions, orientations))
