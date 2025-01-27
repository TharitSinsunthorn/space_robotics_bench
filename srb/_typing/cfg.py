from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from srb.core.env import (
        DirectEnv,
        DirectEnvCfg,
        DirectMarlEnv,
        DirectMarlEnvCfg,
        ManagedEnv,
        ManagedEnvCfg,
    )

AnyEnv: TypeAlias = "DirectEnv | ManagedEnv | DirectMarlEnv"
AnyEnvCfg: TypeAlias = "DirectEnvCfg | ManagedEnvCfg | DirectMarlEnvCfg"
