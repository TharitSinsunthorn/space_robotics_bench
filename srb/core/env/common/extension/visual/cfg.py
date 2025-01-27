from dataclasses import MISSING
from typing import Dict, Literal, Sequence, Tuple

from srb.core.env import InteractiveSceneCfg
from srb.core.sensor import CameraCfg
from srb.utils import configclass


@configclass
class VisualExtCfg:
    ## Subclass requirements
    scene: InteractiveSceneCfg = MISSING  # type: ignore
    agent_rate: int = MISSING  # type: ignore

    ## Camera sensors
    cameras_cfg: Dict[str, CameraCfg] = MISSING  # type: ignore
    camera_resolution: Tuple[int, int] | None = (64, 64)
    camera_framerate: int | None = 0
    camera_data_types: (
        Sequence[
            Literal[
                "rgb",  # same as "rgba",
                "depth",  # same as "distance_to_image_plane",
                "distance_to_camera",
                "normals",
                "motion_vectors",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
                # "instance_segmentation",
                # "instance_id_segmentation",
                # "bounding_box_2d_tight",
                # "bounding_box_2d_tight_fast",
                # "bounding_box_2d_loose",
                # "bounding_box_2d_loose_fast",
                # "bounding_box_3d",
                # "bounding_box_3d_fast",
            ]
        ]
        | None
    ) = ("rgb", "depth")

    def __post_init__(self):
        ## Add camera sensors to the scene
        for key, camera_cfg in self.cameras_cfg.items():
            if self.camera_resolution is not None:
                camera_cfg.width = self.camera_resolution[0]
                camera_cfg.height = self.camera_resolution[1]
            if self.camera_framerate is not None:
                camera_cfg.update_period = (
                    self.camera_framerate
                    if self.camera_framerate > 0
                    else self.agent_rate
                )
            if self.camera_data_types is not None:
                camera_cfg.data_types = self.camera_data_types  # type: ignore
            setattr(self.scene, key, camera_cfg)
