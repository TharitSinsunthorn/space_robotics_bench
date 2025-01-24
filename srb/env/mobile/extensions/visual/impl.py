from typing import Dict, Tuple

import torch

from srb.core.envs import InteractiveScene
from srb.core.sensors import Camera
from srb.utils import image_proc
from srb.utils.str import sanitize_cam_name

from .cfg import VisualMobileRoboticsEnvExtCfg


class VisualMobileRoboticsEnvExt:
    ## Subclass requirements
    common_step_counter: int
    scene: InteractiveScene
    cfg: VisualMobileRoboticsEnvExtCfg

    def __init__(self, cfg: VisualMobileRoboticsEnvExtCfg, **kwargs):
        ## Extract camera sensors from the scene
        self.__cameras: Dict[
            str,  # Name of the output image
            Tuple[
                Camera,  # Camera sensor
                Tuple[float, float],  # Depth range
            ],
        ] = {
            f"image_{sanitize_cam_name(key)}": (
                sensor,
                getattr(cfg.scene, key).spawn.clipping_range,
            )
            for key, sensor in self.scene._sensors.items()
            if isinstance(sensor, Camera)
        }

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        observation = {}
        for image_name, (sensor, depth_range) in self.__cameras.items():
            observation.update(
                image_proc.construct_observation(
                    **image_proc.extract_images(sensor),
                    depth_range=depth_range,
                    image_name=image_name,
                )
            )
        return observation
