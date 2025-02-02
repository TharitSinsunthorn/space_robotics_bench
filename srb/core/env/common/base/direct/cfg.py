import gymnasium
from isaaclab.envs import DirectRLEnvCfg as __DirectRLEnvCfg

from srb.utils.cfg import configclass

from ..cfg import BaseEnvCfg
from ..event_cfg import BaseEventCfg


@configclass
class DirectEnvCfg(BaseEnvCfg, __DirectRLEnvCfg):
    ## Scenario
    seed: int = 0

    ## Events
    events: BaseEventCfg = BaseEventCfg()

    ## Patch isaaclab.envs.DirectRLEnvCfg
    # Disable UI window by default
    ui_window_class_type: type | None = None
    # Ugly hack to gain compatibility with new Isaac Lab
    action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    state_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def __post_init__(self):
        BaseEnvCfg.__post_init__(self)
