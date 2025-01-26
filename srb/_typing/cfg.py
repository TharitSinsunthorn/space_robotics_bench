from typing import TypeAlias

from srb.core.env import DirectEnvCfg, DirectMarlEnvCfg, ManagedEnvCfg

AnyEnvCfg: TypeAlias = DirectEnvCfg | ManagedEnvCfg | DirectMarlEnvCfg
