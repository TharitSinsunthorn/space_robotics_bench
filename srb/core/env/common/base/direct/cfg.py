import gymnasium
from omni.isaac.lab.envs import DirectRLEnvCfg as __DirectRLEnvCfg

from srb.utils.cfg import configclass

from ..cfg import BaseEnvCfg

# # Add this somewhere here
# from srb.core.mdp import randomize_physics_scene_gravity
# reset_gravity = EventTerm(
#     func=randomize_physics_scene_gravity,
#     mode="interval",
#     is_global_time=True,
#     interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
#     params={
#         "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
#         "operation": "add",
#         "distribution": "gaussian",
#     },
# )


@configclass
class DirectEnvCfg(BaseEnvCfg, __DirectRLEnvCfg):
    seed: int = 0

    ## Updated defaults
    # Disable UI window by default
    ui_window_class_type: type | None = None
    # Redundant: spaces are automatically extracted
    num_actions: int = 0
    # Redundant: spaces are automatically extracted
    num_observations: int = 0

    ## Misc
    # Flag that disables the timeout for the environment
    enable_truncation: bool = True

    ## Ugly hack to gain compatibility with new Isaac Lab
    # TODO: Fix spaces in a better way (perhaps reimplement the Env class)
    action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    state_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))
