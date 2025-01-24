import simforge_foundry

from srb.core.actions import SpacecraftActionCfg, SpacecraftActionGroupCfg
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
    action_cfg: SpacecraftActionGroupCfg = SpacecraftActionGroupCfg(
        flight=SpacecraftActionCfg(asset_name="robot", scale=0.1)
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="")
