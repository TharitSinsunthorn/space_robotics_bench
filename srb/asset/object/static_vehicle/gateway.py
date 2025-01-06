import srb.core.sim as sim_utils
import srb.utils.math as math_utils
from srb.core.asset import AssetBaseCfg, Frame, StaticVehicle, Transform
from srb.utils.path import SRB_ASSETS_DIR_SRB_VEHICLE


class Gateway(StaticVehicle):
    ## Model
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/vehicle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_VEHICLE.joinpath("gateway")
            .joinpath("gateway.usdc")
            .as_posix(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    ## Frames
    frame_manipulator_base: Frame = Frame(prim_relpath="base")
    frame_camera_base: Frame = Frame(
        prim_relpath="camera_base",
        offset=Transform(
            translation=(0.21, 0.0, 0.0),
            rotation=math_utils.quat_from_rpy(0.0, 15.0, 0.0),
        ),
    )
    frame_cargo_bay: Frame = Frame(
        prim_relpath="cargo_bay",
        offset=Transform(translation=(-0.6, 0.0, 0.3)),
    )
