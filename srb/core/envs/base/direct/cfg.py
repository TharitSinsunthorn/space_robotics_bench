from dataclasses import MISSING

import gymnasium
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.utils import configclass

from srb.core.asset import Robot
from srb.core.envs.env_cfg import EnvironmentConfig

# # Add this somewhere here
# reset_gravity = EventTerm(
#     func=mdp.randomize_physics_scene_gravity,
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
class BaseEnvCfg(DirectRLEnvCfg):
    """
    Extended version of :class:`omni.isaac.lab.envs.DirectRLEnvCfg`.
    """

    seed: int = 0

    ## Updated defaults
    # Disable UI window by default
    ui_window_class_type: type | None = None
    # Redundant: spaces are automatically extracted
    num_actions: int = 0
    # Redundant: spaces are automatically extracted
    num_observations: int = 0

    ## Environment
    # TODO: Rename to something more sensible
    env_cfg: EnvironmentConfig = EnvironmentConfig()

    robot_cfg: Robot = MISSING  # type: ignore

    ## Misc
    # Flag that disables the timeout for the environment
    enable_truncation: bool = True

    ## Ugly hack to gain compatibility with new Isaac Lab
    # TODO: Fix in a better way
    action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    state_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    # TODO: Tackle this and other variables of the superclass
    rerender_on_reset: bool = True
