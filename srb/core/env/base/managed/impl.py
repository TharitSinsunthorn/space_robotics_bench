from omni.isaac.lab.envs import ManagerBasedRLEnv as __ManagerBasedRLEnv

from srb.core.env.base.managed.cfg import ManagedEnvCfg


class ManagedEnv(__ManagerBasedRLEnv):
    cfg: ManagedEnvCfg
