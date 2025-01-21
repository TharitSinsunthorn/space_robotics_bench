from srb.core import sim as sim_utils
from srb.core.asset import Object, RigidObjectCfg
from srb.core.sim import UsdFileCfg
from srb.utils.path import SRB_ASSETS_DIR_SRB_OBJECT


class SampleTube(Object):
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=UsdFileCfg(
            usd_path=(
                SRB_ASSETS_DIR_SRB_OBJECT.joinpath("sample_tube")
                .joinpath("sample_tube.usdc")
                .as_posix()
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
                mesh_approximation="convexDecomposition"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=1500.0),
        ),
    )
