from omni.isaac.lab.envs import ManagerBasedRLEnvCfg as __ManagerBasedRLEnvCfg

from srb.utils.cfg import configclass

from ..cfg import BaseEnvCfg


@configclass
class ManagedEnvCfg(BaseEnvCfg, __ManagerBasedRLEnvCfg):
    pass
