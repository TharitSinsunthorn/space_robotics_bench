from srb.core.asset import AssetBaseCfg, Frame, StaticVehicle, Transform
from srb.core.sim import CollisionPropertiesCfg, UsdFileCfg
from srb.utils.math import quat_from_rpy
from srb.utils.path import SRB_ASSETS_DIR_SRB_VEHICLE


class Gateway(StaticVehicle):
    ## Model
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/vehicle",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_VEHICLE.joinpath("gateway")
            .joinpath("gateway.usdc")
            .as_posix(),
            collision_props=CollisionPropertiesCfg(),
        ),
    )

    ## Frames
    frame_manipulator_base: Frame = Frame(prim_relpath="base")
    frame_camera_base: Frame = Frame(
        prim_relpath="camera_base",
        offset=Transform(
            translation=(0.21, 0.0, 0.0),
            rotation=quat_from_rpy(0.0, 15.0, 0.0),
        ),
    )
    frame_cargo_bay: Frame = Frame(
        prim_relpath="cargo_bay",
        offset=Transform(translation=(-0.6, 0.0, 0.3)),
    )
