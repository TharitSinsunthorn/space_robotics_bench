from isaaclab.envs import DirectMARLEnvCfg as __DirectMARLEnvCfg

from srb.utils.cfg import configclass

from ..cfg import BaseEnvCfg
from ..event_cfg import BaseEventCfg


@configclass
class DirectMarlEnvCfg(BaseEnvCfg, __DirectMARLEnvCfg):
    ## Scenario
    seed: int = 0

    ## Events
    events: BaseEventCfg = BaseEventCfg()

    def __post_init__(self):
        BaseEnvCfg.__post_init__(self)
