from typing import Any, Dict, Tuple

import srb.core.envs as env_utils
import srb.core.sim as sim_utils
from srb.core.asset import AssetBaseCfg, RigidObjectCfg
from srb.utils import color as color_utils

from .debris import CubesatDebris
from .peg_in_hole_procgen import HoleProcgenCfg, PegProcgenCfg
from .peg_in_hole_profile import HoleProfileCfg, PegProfileCfg, PegProfileShortCfg
from .rock_procgen import LunarRockCfg, MarsRockCfg
from .sample_tube import SampleTubeCfg
from .solar_panel import SolarPanelCfg
from .static_vehicle import *  # noqa: F403


@staticmethod
def object_of_interest_from_env_cfg(
    env_cfg: env_utils.EnvironmentConfig,
    *,
    prim_path: str = "{ENV_REGEX_NS}/object",
    num_assets: int = 1,
    size: Tuple[float, float] = (0.06, 0.06, 0.04),
    spawn_kwargs: Dict[str, Any] = {},
    procgen_seed_offset: int = 0,
    procgen_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> RigidObjectCfg | None:
    spawn = None

    if spawn_kwargs.get("collision_props", None) is None:
        spawn_kwargs["collision_props"] = sim_utils.CollisionPropertiesCfg()
    if spawn_kwargs.get("rigid_props", None) is None:
        spawn_kwargs["rigid_props"] = sim_utils.RigidBodyPropertiesCfg()
    if spawn_kwargs.get("mass_props", None) is None:
        spawn_kwargs["mass_props"] = sim_utils.MassPropertiesCfg(density=2000.0)

    match env_cfg.assets.object.variant:
        case env_utils.AssetVariant.PRIMITIVE:
            if spawn_kwargs.get("visual_material", None) is None:
                spawn_kwargs["visual_material"] = (
                    color_utils.preview_surface_from_env_cfg(env_cfg)
                )

            spawn = sim_utils.MultiShapeCfg(
                size=size,
                shape_cfg=sim_utils.ShapeCfg(**spawn_kwargs),
            )

        case env_utils.AssetVariant.DATASET:
            match env_cfg.domain:
                case env_utils.Domain.MARS:
                    if spawn_kwargs.get("mesh_collision_props", None) is None:
                        spawn_kwargs["mesh_collision_props"] = (
                            sim_utils.MeshCollisionPropertiesCfg(
                                mesh_approximation="sdf",
                            )
                        )
                    spawn = SampleTubeCfg(**spawn_kwargs)
                case _:
                    if spawn_kwargs.get("mesh_collision_props", None) is None:
                        spawn_kwargs["mesh_collision_props"] = (
                            sim_utils.MeshCollisionPropertiesCfg(
                                mesh_approximation="boundingCube"
                            )
                        )
                    if spawn_kwargs.get("visual_material", None) is None:
                        spawn_kwargs["visual_material"] = (
                            color_utils.preview_surface_from_env_cfg(env_cfg)
                        )

                    spawn = sim_utils.MultiAssetSpawnerCfg(
                        assets_cfg=[
                            PegProfileCfg(**spawn_kwargs),
                            PegProfileShortCfg(**spawn_kwargs),
                        ]
                    )

        case env_utils.AssetVariant.PROCEDURAL:
            if spawn_kwargs.get("mesh_collision_props", None) is None:
                spawn_kwargs["mesh_collision_props"] = (
                    sim_utils.MeshCollisionPropertiesCfg(
                        mesh_approximation="sdf",
                    )
                )
            # usd_file_cfg = sim_utils.UsdFileCfg(
            #     usd_path="IGNORED",
            #     **spawn_kwargs,
            # )

            match env_cfg.domain:
                case env_utils.Domain.ORBIT:
                    spawn = CubesatDebris(
                        num_assets=num_assets,
                        seed=env_cfg.seed + procgen_seed_offset,
                        **spawn_kwargs,
                    )

                case env_utils.Domain.MOON:
                    spawn = LunarRockCfg(
                        num_assets=num_assets,
                        seed=env_cfg.seed + procgen_seed_offset,
                        **spawn_kwargs,
                    )

                case env_utils.Domain.MARS:
                    spawn = MarsRockCfg(
                        num_assets=num_assets,
                        seed=env_cfg.seed + procgen_seed_offset,
                        **spawn_kwargs,
                    )
                case _:
                    return None

    if spawn is None:
        raise NotImplementedError
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawn,
        **kwargs,
    )


@staticmethod
def peg_in_hole_from_env_cfg(
    env_cfg: env_utils.EnvironmentConfig,
    *,
    prim_path_peg: str = "{ENV_REGEX_NS}/peg",
    prim_path_hole: str = "{ENV_REGEX_NS}/hole",
    num_assets: int = 1,
    size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    spawn_kwargs_peg: Dict[str, Any] = {},
    spawn_kwargs_hole: Dict[str, Any] = {},
    procgen_seed_offset: int = 0,
    procgen_kwargs_peg: Dict[str, Any] = {},
    procgen_kwargs_hole: Dict[str, Any] = {},
    short_peg: bool = False,
    **kwargs,
) -> Tuple[RigidObjectCfg, AssetBaseCfg] | None:
    spawn_peg = None
    spawn_hole = None

    if spawn_kwargs_peg.get("collision_props", None) is None:
        spawn_kwargs_peg["collision_props"] = sim_utils.CollisionPropertiesCfg()
    if spawn_kwargs_hole.get("collision_props", None) is None:
        spawn_kwargs_hole["collision_props"] = sim_utils.CollisionPropertiesCfg()
    if spawn_kwargs_peg.get("rigid_props", None) is None:
        spawn_kwargs_peg["rigid_props"] = sim_utils.RigidBodyPropertiesCfg()
    if spawn_kwargs_peg.get("mass_props", None) is None:
        spawn_kwargs_peg["mass_props"] = sim_utils.MassPropertiesCfg(density=2000.0)

    match env_cfg.assets.object.variant:
        case env_utils.AssetVariant.DATASET:
            if spawn_kwargs_peg.get("visual_material", None) is None:
                spawn_kwargs_peg["visual_material"] = (
                    color_utils.preview_surface_from_env_cfg(env_cfg)
                )
            if spawn_kwargs_hole.get("visual_material", None) is None:
                spawn_kwargs_hole["visual_material"] = (
                    color_utils.preview_surface_from_env_cfg(env_cfg)
                )
            if spawn_kwargs_peg.get("mesh_collision_props", None) is None:
                spawn_kwargs_peg["mesh_collision_props"] = (
                    sim_utils.MeshCollisionPropertiesCfg(
                        mesh_approximation="boundingCube",
                    )
                )
            spawn_peg = (
                PegProfileShortCfg(**spawn_kwargs_peg)
                if short_peg
                else PegProfileCfg(**spawn_kwargs_peg)
            )
            spawn_hole = HoleProfileCfg(**spawn_kwargs_hole)
        case env_utils.AssetVariant.PROCEDURAL:
            if spawn_kwargs_peg.get("mesh_collision_props", None) is None:
                spawn_kwargs_peg["mesh_collision_props"] = (
                    sim_utils.MeshCollisionPropertiesCfg(
                        mesh_approximation="sdf",
                    )
                )
            spawn_peg = PegProcgenCfg(**spawn_kwargs_peg)
            spawn_hole = HoleProcgenCfg(**spawn_kwargs_hole)

    return RigidObjectCfg(
        prim_path=prim_path_peg,
        spawn=spawn_peg,
        **kwargs,
    ), AssetBaseCfg(
        prim_path=prim_path_hole,
        spawn=spawn_hole,
        **kwargs,
    )


@staticmethod
def solar_panel_from_env_cfg(
    env_cfg: env_utils.EnvironmentConfig,
    *,
    prim_path: str = "{ENV_REGEX_NS}/panel",
    spawn_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> RigidObjectCfg | None:
    spawn = None

    if spawn_kwargs.get("collision_props", None) is None:
        spawn_kwargs["collision_props"] = sim_utils.CollisionPropertiesCfg()
    if spawn_kwargs.get("rigid_props", None) is None:
        spawn_kwargs["rigid_props"] = sim_utils.RigidBodyPropertiesCfg()
    if spawn_kwargs.get("mass_props", None) is None:
        spawn_kwargs["mass_props"] = sim_utils.MassPropertiesCfg(density=1000.0)

    if spawn_kwargs.get("mesh_collision_props", None) is None:
        spawn_kwargs["mesh_collision_props"] = sim_utils.MeshCollisionPropertiesCfg(
            mesh_approximation="sdf",
        )

    spawn = SolarPanelCfg(**spawn_kwargs)

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawn,
        **kwargs,
    )
