from omni.isaac.lab.assets.rigid_object import RigidObjectCfg
from simforge.isaaclab import SimforgeAssetCfg
from simforge_foundry.models import Cubesat

import space_robotics_bench.core.assets as asset_utils
import space_robotics_bench.core.sim as sim_utils
from space_robotics_bench.core.actions import (
    SpacecraftActionCfg,
    SpacecraftActionGroupCfg,
)


def cubesat_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/robot",
    asset_name: str = "robot",
    action_scale: float = 0.05,
    **kwargs,
) -> asset_utils.SpacecraftCfg:
    frame_base = "TODO"
    return asset_utils.SpacecraftCfg(
        ## Model
        asset_cfg=RigidObjectCfg(
            spawn=SimforgeAssetCfg(
                assets=[Cubesat()],
                num_assets=16,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
                #     mesh_approximation="sdf",
                # ),
                activate_contact_sensors=False,
                random_choice=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        ).replace(prim_path=prim_path, **kwargs),
        ## Actions
        action_cfg=SpacecraftActionGroupCfg(
            flight=SpacecraftActionCfg(
                asset_name=asset_name,
                scale=action_scale,
            )
        ),
        ## Frames
        frame_base=asset_utils.FrameCfg(
            prim_relpath=frame_base,
        ),
    )
