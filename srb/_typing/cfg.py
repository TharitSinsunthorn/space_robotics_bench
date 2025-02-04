from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from srb.core.env import (
        BaseEnvCfg,
        DirectEnv,
        DirectEnvCfg,
        DirectMarlEnv,
        DirectMarlEnvCfg,
        ManagedEnv,
        ManagedEnvCfg,
    )

AnyEnv: TypeAlias = "DirectEnv | ManagedEnv | DirectMarlEnv"
AnyEnvCfg: TypeAlias = "BaseEnvCfg | DirectEnvCfg | ManagedEnvCfg | DirectMarlEnvCfg"
