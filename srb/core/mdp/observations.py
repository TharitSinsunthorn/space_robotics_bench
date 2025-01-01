from typing import TYPE_CHECKING

import torch
from omni.isaac.lab.managers import SceneEntityCfg

from srb.core.asset import Articulation

if TYPE_CHECKING:
    from srb.core.envs import BaseEnv


def body_incoming_wrench_mean(
    env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    link_incoming_forces = asset.root_physx_view.get_link_incoming_joint_force()[
        :, asset_cfg.body_ids
    ]
    return link_incoming_forces.mean(dim=1)
