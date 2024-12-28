from os import path
from typing import Any, Dict

import space_robotics_bench.core.asset as asset_utils
import space_robotics_bench.core.sim as sim_utils
import space_robotics_bench.utils.math as math_utils
from space_robotics_bench.paths import SRB_ASSETS_DIR_SRB_VEHICLE


def construction_rover_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/vehicle",
    spawn_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> asset_utils.VehicleCfg:
    if spawn_kwargs.get("collision_props") is None:
        spawn_kwargs["collision_props"] = sim_utils.CollisionPropertiesCfg()

    return asset_utils.VehicleCfg(
        ## Model
        asset_cfg=asset_utils.AssetBaseCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=path.join(
                    SRB_ASSETS_DIR_SRB_VEHICLE,
                    "construction_rover",
                    "construction_rover.usdc",
                ),
                **spawn_kwargs,
            ),
            **kwargs,
        ),
        ## Frames
        frame_manipulator_base=asset_utils.Frame(
            prim_relpath="manipulator_base",
            offset=asset_utils.Transform(
                translation=(0.0, 0.0, 0.25),
            ),
        ),
        frame_camera_base=asset_utils.Frame(
            prim_relpath="camera_base",
            offset=asset_utils.Transform(
                translation=(0.21, 0.0, 0.0),
                rotation=math_utils.quat_from_rpy(0.0, 45.0, 0.0),
            ),
        ),
        frame_cargo_bay=asset_utils.Frame(
            prim_relpath="cargo_bay",
            offset=asset_utils.Transform(
                translation=(-0.6, 0.0, 0.3),
            ),
        ),
        ## Properties
        height=0.25,
    )
