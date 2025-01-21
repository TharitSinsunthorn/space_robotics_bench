from simforge_foundry import geometry as sf_geometry

from srb.core import sim as sim_utils
from srb.core.asset import AssetBaseCfg, Object, RigidObjectCfg
from srb.core.sim import SimforgeAssetCfg


class Peg(Object):
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/peg",
        spawn=SimforgeAssetCfg(
            assets=[sf_geometry.PegGeo()],
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
                mesh_approximation="sdf"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
        ),
    )


# TODO: Consider making it a kinematic rigid body object
class Hole(Object):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/hole",
        spawn=SimforgeAssetCfg(
            assets=[sf_geometry.HoleGeo()],
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
            #     mesh_approximation="sdf"
            # ),
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            # mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
        ),
    )
