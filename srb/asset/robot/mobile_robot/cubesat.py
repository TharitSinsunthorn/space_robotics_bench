from omni.isaac.lab.assets.rigid_object import RigidObjectCfg
from simforge_foundry import model as sf_model

from srb.core import sim as sim_utils
from srb.core.actions import SpacecraftActionCfg, SpacecraftActionGroupCfg
from srb.core.asset import Frame, Spacecraft


class Cubesat(Spacecraft):
    ## Model
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.SimforgeAssetCfg(
            assets=[sf_model.Cubesat()],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
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
