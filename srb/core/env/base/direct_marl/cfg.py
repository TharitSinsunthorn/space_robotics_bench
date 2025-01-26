from omni.isaac.lab.envs import DirectMARLEnvCfg as __DirectMARLEnvCfg

from srb.core.env.base.cfg import BaseEnvCfg
from srb.utils import configclass


@configclass
class DirectMarlEnvCfg(BaseEnvCfg, __DirectMARLEnvCfg):
    pass
