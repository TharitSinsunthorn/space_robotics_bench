from os import path
from typing import Any, Dict, List

from omni.isaac.lab.utils import configclass

import space_robotics_bench.core.sim as sim_utils
from space_robotics_bench.paths import SRB_ASSETS_DIR_SRB_OBJECT
from space_robotics_bench.utils.path import abs_listdir


@configclass
class PegProcgenCfg(sim_utils.BlenderNodesAssetCfg):
    name: str = "peg"
    autorun_scripts: List[str] = abs_listdir(
        path.join(SRB_ASSETS_DIR_SRB_OBJECT, "peg_in_hole_procgen")
    )

    # Geometry
    geometry_nodes: Dict[str, Dict[str, Any]] = {"Peg": {}}
    decimate_face_count: Optional[int] = None
    decimate_angle_limit: Optional[float] = None

    # Material
    material: Optional[str] = "Metal"
    texture_resolution: int = 512


@configclass
class HoleProcgenCfg(sim_utils.BlenderNodesAssetCfg):
    name: str = "hole"
    autorun_scripts: List[str] = abs_listdir(
        path.join(SRB_ASSETS_DIR_SRB_OBJECT, "peg_in_hole_procgen")
    )

    export_kwargs: Dict[str, Any] = {"triangulate_meshes": True}

    # Geometry
    geometry_nodes: Dict[str, Dict[str, Any]] = {"BlankModule": {}, "Hole": {}}
    decimate_face_count: Optional[int] = None
    decimate_angle_limit: Optional[float] = None

    # Material
    material: Optional[str] = "Metal"
    texture_resolution: int = 1024
