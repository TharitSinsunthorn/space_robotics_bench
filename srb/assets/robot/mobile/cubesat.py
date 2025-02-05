import simforge_foundry

from srb.core.action import BodyVelocity, BodyVelocityActionCfg
from srb.core.asset import Frame, RigidObjectCfg, Spacecraft
from srb.core.sim import (
    CollisionPropertiesCfg,
    MeshCollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
    SimforgeAssetCfg,
)


class Cubesat(Spacecraft):
    ## Model
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=SimforgeAssetCfg(
            assets=[simforge_foundry.Cubesat()],
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            collision_props=CollisionPropertiesCfg(),
            mesh_collision_props=MeshCollisionPropertiesCfg(
                mesh_approximation="meshSimplification"
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    ## Actions
    action_cfg: BodyVelocity = BodyVelocity(
        vel=BodyVelocityActionCfg(asset_name="robot", scale=0.01)
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="")
