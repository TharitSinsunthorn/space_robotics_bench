from typing import TYPE_CHECKING

import torch

from srb.core.asset import Articulation
from srb.core.manager import SceneEntityCfg

if TYPE_CHECKING:
    from srb.core.env import DirectEnv


def body_incoming_wrench_mean(
    env: "DirectEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    link_incoming_forces = asset.root_physx_view.get_link_incoming_joint_force()[
        :, asset_cfg.body_ids
    ]
    return link_incoming_forces.mean(dim=1)
