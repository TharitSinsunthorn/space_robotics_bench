from os import path

import space_robotics_bench.core.sim as sim_utils
import space_robotics_bench.utils.math as math_utils
from space_robotics_bench.core.asset import (
    AssetBaseCfg,
    Frame,
    StaticVehicle,
    Transform,
)
from space_robotics_bench.paths import SRB_ASSETS_DIR_SRB_VEHICLE


class ConstructionRover(StaticVehicle):
    ## Model
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/vehicle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=path.join(
                SRB_ASSETS_DIR_SRB_VEHICLE,
                "construction_rover",
                "construction_rover.usdc",
            ),
        ),
    )

    ## Frames
    frame_manipulator_base: Frame = Frame(
        prim_relpath="manipulator_base",
        offset=Transform(
            translation=(0.0, 0.0, 0.25),
        ),
    )
    frame_camera_base: Frame = Frame(
        prim_relpath="camera_base",
        offset=Transform(
            translation=(0.21, 0.0, 0.0),
            rotation=math_utils.quat_from_rpy(0.0, 45.0, 0.0),
        ),
    )
    frame_cargo_bay: Frame = Frame(
        prim_relpath="cargo_bay",
        offset=Transform(
            translation=(-0.6, 0.0, 0.3),
        ),
    )

    # height: float = 0.25
