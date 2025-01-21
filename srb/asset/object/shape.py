from srb.core import sim as sim_utils
from srb.core.asset import Object, RigidObjectCfg
from srb.core.sim import MultiShapeSpawnerCfg


class RandomShape(Object):
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/shape",
        spawn=MultiShapeSpawnerCfg(
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
        ),
    )
