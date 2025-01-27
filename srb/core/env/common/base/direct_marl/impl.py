from omni.isaac.lab.envs import DirectMARLEnv as __DirectMARLEnv

from .cfg import DirectMarlEnvCfg


class DirectMarlEnv(__DirectMARLEnv):
    cfg: DirectMarlEnvCfg
