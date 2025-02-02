from isaaclab.envs import ManagerBasedRLEnvCfg as __ManagerBasedRLEnvCfg

from srb.utils.cfg import configclass

from ..cfg import BaseEnvCfg
from ..event_cfg import BaseEventCfg


@configclass
class ManagedEnvCfg(BaseEnvCfg, __ManagerBasedRLEnvCfg):
    ## Scenario
    seed: int = 0

    ## Events
    events: BaseEventCfg = BaseEventCfg()

    def __post_init__(self):
        BaseEnvCfg.__post_init__(self)
