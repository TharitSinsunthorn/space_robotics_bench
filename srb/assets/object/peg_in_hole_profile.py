from srb.core.asset import Object, RigidObjectCfg
from srb.core.sim import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    MeshCollisionPropertiesCfg,
    PreviewSurfaceCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)
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
            collision_props=CollisionPropertiesCfg(),
            mesh_collision_props=MeshCollisionPropertiesCfg(
                mesh_approximation="boundingCube"
            ),
            rigid_props=RigidBodyPropertiesCfg(),
            mass_props=MassPropertiesCfg(density=1000.0),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
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
            collision_props=CollisionPropertiesCfg(),
            mesh_collision_props=MeshCollisionPropertiesCfg(
                mesh_approximation="boundingCube"
            ),
            rigid_props=RigidBodyPropertiesCfg(),
            mass_props=MassPropertiesCfg(density=1000.0),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )


class ProfileHole(Object):
    asset_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=UsdFileCfg(
            usd_path=(
                SRB_ASSETS_DIR_SRB_OBJECT.joinpath("peg_in_hole_profile")
                .joinpath("hole.usdc")
                .as_posix()
            ),
            collision_props=CollisionPropertiesCfg(),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )
