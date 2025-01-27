from typing import Dict, Sequence, Tuple

import torch

from srb.core.env import InteractiveScene
from srb.utils.str import sanitize_cam_name

from .cfg import VisualExtCfg


class VisualExt:
    ## Subclass requirements
    scene: InteractiveScene

    def __init__(self, cfg: VisualExtCfg, **kwargs):
        self.__cameras = [
            (
                self.scene.sensors[camera_key],
                f"image_{sanitize_cam_name(camera_key)}",
                cfg.cameras_cfg[camera_key].data_types,
                cfg.cameras_cfg[camera_key].spawn.clipping_range,  # type: ignore
            )
            for camera_key in cfg.cameras_cfg.keys()
        ]

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {
            image_key: image
            for camera, image_basename, data_types, clipping_range in self.__cameras
            for image_key, image in construct_observation(
                image_basename=image_basename,
                data_types=data_types,
                clipping_range=clipping_range,  # type: ignore
                **camera.data.output,
            ).items()
        }


def construct_observation(
    *,
    image_basename: str,
    data_types: Sequence[str],
    clipping_range: Tuple[float, float],
    **kwargs,
) -> Dict[str, torch.Tensor]:
    return {
        f"{image_basename}_{data_type}": _PROCESSORS[data_type](
            kwargs[data_type], clipping_range
        )
        if data_type == "depth"
        else _PROCESSORS[data_type](kwargs[data_type])
        for data_type in data_types
    }


@torch.jit.script
def process_rgb(image: torch.Tensor) -> torch.Tensor:
    return image[..., :3].to(torch.float32) / 255.0


@torch.jit.script
def process_depth(
    image: torch.Tensor,
    clipping_range: Tuple[float, float],
) -> torch.Tensor:
    return (
        image.nan_to_num(
            nan=clipping_range[1], posinf=clipping_range[1], neginf=clipping_range[1]
        ).clamp(clipping_range[0], clipping_range[1])
        - clipping_range[0]
    ) / (clipping_range[1] - clipping_range[0])


_PROCESSORS = {
    "rgb": process_rgb,
    "depth": process_depth,
}
