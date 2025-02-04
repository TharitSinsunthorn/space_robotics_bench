from typing import TYPE_CHECKING, Tuple

import torch
from pydantic import BaseModel
from simforge import TexResConfig

from srb import assets
from srb.core.asset import AssetVariant, RigidObjectCfg
from srb.core.domain import Domain
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import reset_root_state_uniform

if TYPE_CHECKING:
    from srb._typing import AnyEnvCfg


class SampleCfg(BaseModel, arbitrary_types_allowed=True):
    ## Model
    asset_cfg: RigidObjectCfg

    ## Randomization
    state_randomizer: EventTermCfg


def sample_cfg(
    cfg: "AnyEnvCfg",
    *,
    seed: int,
    num_assets: int,
    prim_path: str = "{ENV_REGEX_NS}/sample",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    scale: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    texture_resolution: TexResConfig | None = None,
    **kwargs,
) -> SampleCfg:
    pose_range = {
        "x": (-0.2, 0.2),
        "y": (-0.3, 0.3),
        "roll": (-torch.pi, torch.pi),
        "pitch": (-torch.pi, torch.pi),
        "yaw": (-torch.pi, torch.pi),
    }

    match cfg.obj:
        case AssetVariant.PRIMITIVE:
            pose_range["z"] = (0.1, 0.1)
        case AssetVariant.DATASET:
            match cfg.domain:
                case Domain.MARS:
                    pose_range.update(
                        {
                            "z": (0.05, 0.05),
                            "roll": (torch.pi / 7, torch.pi / 7),
                            "pitch": (
                                87.5 * torch.pi / 180,
                                87.5 * torch.pi / 180,
                            ),
                            "yaw": (-torch.pi, torch.pi),
                        }
                    )
                case _:
                    pose_range["z"] = (0.07, 0.07)
        case _:
            pose_range["z"] = (0.06, 0.06)

    sample_cfg = assets.rigid_object_from_cfg(
        cfg,
        seed=seed,
        num_assets=num_assets,
        prim_path=prim_path,
        scale=scale,
        texture_resolution=texture_resolution,
        **kwargs,
    )

    return SampleCfg(
        asset_cfg=sample_cfg,
        state_randomizer=EventTermCfg(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": asset_cfg,
                "pose_range": pose_range,
                "velocity_range": {},
            },
        ),
    )
