from omni.isaac.lab.envs import DirectMARLEnv as __DirectMARLEnv

from srb.core.env.base.direct_marl.cfg import DirectMarlEnvCfg


class DirectMarlEnv(__DirectMARLEnv):
    cfg: DirectMarlEnvCfg
