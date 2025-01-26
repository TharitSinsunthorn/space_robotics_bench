from srb.core.asset import Object, Robot, Terrain
from srb.core.env.base.enums import AssetVariant, Domain
from srb.utils import configclass


@configclass
class BaseEnvCfg:
    ## Misc
    # Flag that disables the timeout for the environment
    enable_truncation: bool = True

    ## Scenario
    domain: Domain = Domain.MOON

    ## Assets
    robot: Robot | AssetVariant | None = AssetVariant.DATASET
    obj: Object | AssetVariant | None = AssetVariant.PROCEDURAL
    terrain: Terrain | AssetVariant | None = AssetVariant.PROCEDURAL
