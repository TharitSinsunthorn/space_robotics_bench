from typing import TypeAlias

from srb.core.envs import DirectEnvCfg, DirectMarlEnvCfg, ManagedEnvCfg

AnyEnvCfg: TypeAlias = DirectEnvCfg | ManagedEnvCfg | DirectMarlEnvCfg
