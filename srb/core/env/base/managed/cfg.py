from omni.isaac.lab.envs import ManagerBasedRLEnvCfg as __ManagerBasedRLEnvCfg

from srb.core.env.base.cfg import BaseEnvCfg
from srb.utils import configclass


@configclass
class ManagedEnvCfg(BaseEnvCfg, __ManagerBasedRLEnvCfg):
    pass
