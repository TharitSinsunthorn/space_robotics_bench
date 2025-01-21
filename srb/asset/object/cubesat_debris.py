from simforge_foundry import model as sf_model

from srb.core import sim as sim_utils
from srb.core.asset import Object, RigidObjectCfg
from srb.core.sim import SimforgeAssetCfg


class CubesatDebris(Object):
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=SimforgeAssetCfg(
            assets=[sf_model.Cubesat()],
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
                mesh_approximation="sdf"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
        ),
    )
