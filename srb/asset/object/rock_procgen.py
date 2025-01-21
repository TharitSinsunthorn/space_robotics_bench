from simforge_foundry import model as sf_model

from srb.core import sim as sim_utils
from srb.core.asset import Object, RigidObjectCfg
from srb.core.sim import SimforgeAssetCfg


class LunarRock(Object):
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=SimforgeAssetCfg(
            assets=[sf_model.MoonRock()],
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
                mesh_approximation="convexDecomposition"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=2000.0),
        ),
    )


class MarsRock(Object):
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=SimforgeAssetCfg(
            assets=[sf_model.MarsRock()],
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
                mesh_approximation="convexDecomposition"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=2000.0),
        ),
    )
