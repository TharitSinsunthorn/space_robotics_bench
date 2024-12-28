from omni.isaac.lab.assets.rigid_object import RigidObjectCfg
from simforge_foundry.models import Cubesat

import space_robotics_bench.core.sim as sim_utils
from space_robotics_bench.core.actions import (
    SpacecraftActionCfg,
    SpacecraftActionGroupCfg,
)
from space_robotics_bench.core.asset import Frame, Spacecraft


class CubeSat(Spacecraft):
    ## Model
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.SimforgeAssetCfg(
            assets=[Cubesat()],
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
        flight=SpacecraftActionCfg(
            asset_name="robot",
            scale=0.1,
        )
    )

    ## Frames
    frame_base: Frame = Frame(
        prim_relpath="",
    )
