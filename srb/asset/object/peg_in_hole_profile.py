from srb.core import sim as sim_utils
from srb.core.asset import AssetBaseCfg, Object, RigidObjectCfg
from srb.core.sim import UsdFileCfg
from srb.utils.path import SRB_ASSETS_DIR_SRB_OBJECT


class ProfilePeg(Object):
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=UsdFileCfg(
            usd_path=(
                SRB_ASSETS_DIR_SRB_OBJECT.joinpath("peg_in_hole_profile")
                .joinpath("profile.usdc")
                .as_posix()
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
                mesh_approximation="boundingCube"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )


class ShortProfilePeg(Object):
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=UsdFileCfg(
            usd_path=(
                SRB_ASSETS_DIR_SRB_OBJECT.joinpath("peg_in_hole_profile")
                .joinpath("profile_short.usdc")
                .as_posix()
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
                mesh_approximation="boundingCube"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )


# TODO: Consider making it a kinematic rigid body object
class ProfileHole(Object):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=UsdFileCfg(
            usd_path=(
                SRB_ASSETS_DIR_SRB_OBJECT.joinpath("peg_in_hole_profile")
                .joinpath("hole.usdc")
                .as_posix()
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
            #     mesh_approximation="sdf"
            # ),
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            # mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )
