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
    from datetime import datetime

    import gymnasium
    import numpy as np
    from omni.isaac.kit import SimulationApp
    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
    from sb3_contrib import TQC, RecurrentPPO
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecNormalize

    import srb  # noqa: F401
    from srb.integrations.sb3.wrapper import Sb3VecEnvWrapper, process_sb3_cfg
    from srb.utils.parsing import load_cfg_from_registry, parse_task_cfg

    ## Extract simulation app
    _sim_app: SimulationApp = launcher.app

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

    # override configuration with command line arguments
    if args.seed is not None:
        agent_cfg["seed"] = args.seed

    # directory for logging into
    log_dir = os.path.join(
        "logs",
        FRAMEWORK_NAME,
        args.algo,
        args.task,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "task.yaml"), task_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "task.pkl"), task_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create isaac environment
    env = gymnasium.make(
        args.task, cfg=task_cfg, render_mode="rgb_array" if args.video else None
    )
    # wrap for video recording
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gymnasium.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)
    # set the seed
    env.seed(seed=agent_cfg["seed"])

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

    # create agent from stable baselines
    if args.algo == "ppo":
        agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    elif args.algo == "ppo_lstm":
        agent = RecurrentPPO(policy_arch, env, verbose=1, **agent_cfg)
    elif args.algo == "sac":
        agent = SAC(policy_arch, env, verbose=1, **agent_cfg)
    elif args.algo == "tqc":
        agent = TQC(policy_arch, env, verbose=1, **agent_cfg)
    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2
    )
    # train the agent
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    # save the final model
    agent.save(os.path.join(log_dir, "model"))

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
        help="Name of the algorithm\n(ppo, sac, ppo_lstm)",
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
