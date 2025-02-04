from typing import TYPE_CHECKING, Any, Dict, Tuple

from pydantic import BaseModel, NonNegativeInt
from simforge import TexResConfig

from srb import assets
from srb.core.asset import RigidObjectCfg

if TYPE_CHECKING:
    from srb._typing import AnyEnvCfg


class PegCfg(BaseModel, arbitrary_types_allowed=True):
    ## Model
    asset_cfg: RigidObjectCfg

    ## Geometry
    offset_pos_ends: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]

    ## Rotational symmetry of the peg represented as integer
    #  0: Circle (infinite symmetry)
    #  1: No symmetry (exactly one fit)
    #  n: n-fold symmetry (360/n deg between each symmetry)
    rot_symmetry_n: NonNegativeInt = 1


class HoleCfg(BaseModel):
    ## Model
    asset_cfg: RigidObjectCfg

    ## Geometry
    offset_pos_bottom: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_pos_entrance: Tuple[float, float, float]


class PegInHoleCfg(BaseModel):
    peg: PegCfg
    hole: HoleCfg


def peg_in_hole_from_cfg(
    cfg: "AnyEnvCfg",
    *,
    seed: int,
    num_assets: int,
    prim_path_peg: str = "{ENV_REGEX_NS}/peg",
    prim_path_hole: str = "{ENV_REGEX_NS}/hole",
    scale: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    texture_resolution: TexResConfig | None = None,
    short_peg: bool = False,
    peg_kwargs: Dict[str, Any],
    hole_kwargs: Dict[str, Any],
    **kwargs,
) -> Tuple[RigidObjectCfg, RigidObjectCfg]:
    # TODO: Fix procedural peg-in-hole module
    # match cfg.obj:
    #     case AssetVariant.DATASET:
    #         peg_cfg = (
    #             assets.ShortProfilePeg() if short_peg else assets.ProfilePeg()
    #         ).asset_cfg
    #         hole_cfg = assets.ProfileHole().asset_cfg

    #     case AssetVariant.PROCEDURAL:
    #         peg_cfg = assets.Peg(
    #             scale=scale, texture_resolution=texture_resolution
    #         ).asset_cfg
    #         peg_cfg.spawn.seed = seed  # type: ignore
    #         peg_cfg.spawn.num_assets = num_assets  # type: ignore

    #         hole_cfg = assets.Hole(
    #             scale=scale, texture_resolution=texture_resolution
    #         ).asset_cfg
    #         hole_cfg.spawn.seed = seed  # type: ignore
    #         hole_cfg.spawn.num_assets = num_assets  # type: ignore
    peg_cfg = (assets.ShortProfilePeg() if short_peg else assets.ProfilePeg()).asset_cfg
    hole_cfg = assets.ProfileHole().asset_cfg

    peg_cfg.prim_path = prim_path_peg
    peg_kwargs.update(**kwargs)
    peg_cfg.spawn.replace(**peg_kwargs)  # type: ignore

    hole_cfg.prim_path = prim_path_hole
    hole_kwargs.update(**kwargs)
    hole_cfg.spawn.replace(**hole_kwargs)  # type: ignore

    return peg_cfg, hole_cfg


def peg_and_hole_cfg(
    env_cfg: "AnyEnvCfg",
    *,
    seed: int,
    init_state: RigidObjectCfg.InitialStateCfg,
    num_assets: int = 1,
    prim_path_peg: str = "{ENV_REGEX_NS}/peg",
    prim_path_hole: str = "{ENV_REGEX_NS}/hole",
    scale: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    peg_kwargs: Dict[str, Any] = {},
    hole_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> PegInHoleCfg:
    rot_symmetry_n = 4
    offset_pos_ends = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.2))
    offset_pos_entrance = (0.0, 0.0, 0.02)

    peg_cfg, hole_cfg = peg_in_hole_from_cfg(
        env_cfg,
        seed=seed,
        num_assets=num_assets,
        prim_path_peg=prim_path_peg,
        prim_path_hole=prim_path_hole,
        scale=scale,
        peg_kwargs=peg_kwargs,
        hole_kwargs=hole_kwargs,
        **kwargs,
    )

    peg_cfg.init_state = init_state
    hole_cfg.init_state = init_state

    return PegInHoleCfg(
        peg=PegCfg(
            asset_cfg=peg_cfg,
            offset_pos_ends=offset_pos_ends,
            rot_symmetry_n=rot_symmetry_n,
        ),
        hole=HoleCfg(
            asset_cfg=hole_cfg,
            offset_pos_entrance=offset_pos_entrance,
        ),
    )
