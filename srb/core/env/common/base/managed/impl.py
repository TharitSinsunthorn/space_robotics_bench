from omni.isaac.lab.envs import ManagerBasedRLEnv as __ManagerBasedRLEnv

from .cfg import ManagedEnvCfg


class ManagedEnv(__ManagerBasedRLEnv):
    cfg: ManagedEnvCfg
