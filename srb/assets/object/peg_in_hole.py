import simforge_foundry

from srb.core.asset import Object, RigidObjectCfg
from srb.core.sim import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    MeshCollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
    SimforgeAssetCfg,
)


class Peg(Object):
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/peg",
        spawn=SimforgeAssetCfg(
            assets=[simforge_foundry.PegGeo()],
            collision_props=CollisionPropertiesCfg(),
            mesh_collision_props=MeshCollisionPropertiesCfg(mesh_approximation="sdf"),
            rigid_props=RigidBodyPropertiesCfg(),
            mass_props=MassPropertiesCfg(density=1000.0),
        ),
    )


class Hole(Object):
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/hole",
        spawn=SimforgeAssetCfg(
            assets=[simforge_foundry.HoleGeo()],
            collision_props=CollisionPropertiesCfg(),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
