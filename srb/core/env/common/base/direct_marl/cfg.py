from omni.isaac.lab.envs import DirectMARLEnvCfg as __DirectMARLEnvCfg

from srb.utils.cfg import configclass

from ..cfg import BaseEnvCfg


@configclass
class DirectMarlEnvCfg(BaseEnvCfg, __DirectMARLEnvCfg):
    pass
