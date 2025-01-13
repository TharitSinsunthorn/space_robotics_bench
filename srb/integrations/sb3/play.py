#!/root/isaac-sim/python.sh

import os
import sys

from omni.isaac.lab.app import AppLauncher

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from _cli_utils import add_default_cli_args, argparse, launch_app, shutdown_app

FRAMEWORK_NAME = "sb3"
ALGO_CFG_ENTRYPOINT_KEY = f"{FRAMEWORK_NAME}_{{ALGO_NAME}}_cfg"


def main(launcher: AppLauncher, args: argparse.Namespace):
    import gymnasium
    import numpy as np
    import torch
    from omni.isaac.kit import SimulationApp
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    import srb  # noqa: F401
    from srb.integrations.sb3.wrapper import Sb3VecEnvWrapper, process_sb3_cfg
    from srb.utils.parsing import (
        get_checkpoint_path,
        load_cfg_from_registry,
        parse_task_cfg,
    )

    ## Extract simulation app
    sim_app: SimulationApp = launcher.app

    # parse configuration
    task_cfg = parse_task_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(
        args.task, ALGO_CFG_ENTRYPOINT_KEY.format(ALGO_NAME=args.algo)
    )
    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)

    # create isaac environment
    env = gymnasium.make(args.task, cfg=task_cfg)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    # normalize environment (if needed)
    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg
            and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg
            and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # directory for logging into
    log_root_path = os.path.join("logs", FRAMEWORK_NAME, args.algo, args.task)
    log_root_path = os.path.abspath(log_root_path)
    # check checkpoint is valid
    if args.checkpoint is None:
        if args.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args.checkpoint
    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = PPO.load(checkpoint_path, env, print_system_info=True)

    # reset environment
    obs = env.reset()
    # simulate environment
    while sim_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, _ = agent.predict(obs, deterministic=True)
            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


### Helper functions ###
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    algorithm_group = parser.add_argument_group(
        "algorithm arguments",
        description="Arguments for algorithm.",
    )
    algorithm_group.add_argument(
        "--algo",
        type=str,
        default="ppo",
        help="Name of the algorithm\n(ppo, sac)",
    )
    add_default_cli_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_cli_args()

    # Launch the app
    launcher = launch_app(args)

    # Run the main function
    main(launcher=launcher, args=args)

    # Shutdown the app
    shutdown_app(launcher)
