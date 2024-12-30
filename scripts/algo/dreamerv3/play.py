#!/root/isaac-sim/python.sh

import os
import sys

from omni.isaac.lab.app import AppLauncher

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from _cli_utils import add_default_cli_args, argparse, launch_app, shutdown_app

ALGO_NAME = "dreamerv3"
ALGO_CFG_ENTRYPOINT_KEY = f"{ALGO_NAME}_cfg"


def main(launcher: AppLauncher, args: argparse.Namespace):
    import re
    from collections import defaultdict

    import dreamerv3
    import embodied
    import gymnasium
    import numpy as np
    from omni.isaac.kit import SimulationApp
    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

    import space_robotics_bench  # noqa: F401
    from space_robotics_bench.core.sim import SimulationContext
    from space_robotics_bench.core.wrappers.dreamerv3 import (
        DriverParallelEnv,
        EmbodiedEnvWrapper,
        process_dreamerv3_cfg,
    )
    from space_robotics_bench.utils.parsing import (
        create_eval_logdir_path,
        get_checkpoint_path,
        load_cfg_from_registry,
        parse_task_cfg,
    )

    ## Extract simulation app
    _sim_app: SimulationApp = launcher.app

    ## Configuration
    task_cfg = parse_task_cfg(
        task_name=args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args.task, ALGO_CFG_ENTRYPOINT_KEY)

    # Allow overriding the seed
    if args.seed is not None:
        agent_cfg["seed"] = args.seed

    # Get path to checkpoint
    checkpoint_path = get_checkpoint_path(
        ALGO_NAME,
        args.task,
        checkpoint=args.checkpoint,
    )

    # Directory for logging
    logdir = create_eval_logdir_path(checkpoint_path)

    dump_yaml(os.path.join(logdir, "params", "task.yaml"), task_cfg)
    dump_pickle(os.path.join(logdir, "params", "task.pkl"), task_cfg)

    dump_yaml(os.path.join(logdir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(logdir, "params", "agent.pkl"), agent_cfg)

    # Post-process agent configuration
    agent_cfg = process_dreamerv3_cfg(
        agent_cfg,
        logdir=logdir,
        num_envs=args.num_envs,
        task_name=args.task,
        model_size=args.model_size,
    )

    def make_env(config):
        ## Create the environment
        env = gymnasium.make(id=args.task, cfg=task_cfg, render_mode="rgb_array")

        ## Initialize the environment
        _observation, _info = env.reset()
        # Render a couple of frames to help with stabilization of rendered images
        sim_context: SimulationContext = SimulationContext.instance()
        for _ in range(64):
            sim_context.render()

        ## Add wrapper for video recording (if enabled)
        if args.video:
            video_kwargs = {
                "video_folder": os.path.join(logdir, "videos"),
                "step_trigger": lambda step: step % args.video_interval == 0,
                "video_length": args.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gymnasium.wrappers.RecordVideo(env, **video_kwargs)

        ## Add wrapper for DreamerV3
        env = EmbodiedEnvWrapper(env)

        ## Seed the environment
        env.seed(seed=config["seed"] if config["seed"] is not None else 42)

        return env

    def make_agent(config, env):
        if config.random_agent:
            agent = embodied.RandomAgent(env.obs_space, env.act_space)
        else:
            agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
        return agent

    def make_logger(config):
        logdir = embodied.Path(config.logdir)
        multiplier = config.env.get(config.task.split("_")[0], {}).get("repeat", 1)
        return embodied.Logger(
            embodied.Counter(),
            [
                embodied.logger.TerminalOutput(config.filter, "Agent"),
                embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
                embodied.logger.JSONLOutput(logdir, "scores.jsonl", "episode/score"),
                embodied.logger.TensorBoardOutput(
                    logdir,
                    config.run.log_video_fps,
                    config.tensorboard_videos,
                    parallel=False,
                ),
                # embodied.logger.WandbOutput(logdir.name, config=config),
            ],
            multiplier,
        )

    run_args = embodied.Config(
        **agent_cfg.run,
        logdir=logdir,
        batch_length_eval=agent_cfg.batch_length_eval,
        batch_length=agent_cfg.batch_length,
        batch_size=agent_cfg.batch_size,
        replay_context=agent_cfg.replay_context,
    )

    # Note: Everything below is a modified replacement for `embodied.run.eval_only`
    # - 1 env is created
    # - DriverParallelEnv is used to run the agent in parallel

    env = make_env(agent_cfg)
    agent = make_agent(agent_cfg, env)
    logger = make_logger(agent_cfg)

    logdir = embodied.Path(run_args.logdir)
    logdir.mkdir()
    print("Logdir", logdir)
    step = logger.step
    usage = embodied.Usage(**run_args.usage)
    agg = embodied.Agg()
    epstats = embodied.Agg()
    episodes = defaultdict(embodied.Agg)
    should_log = embodied.when.Clock(run_args.log_every)
    policy_fps = embodied.FPS()

    @embodied.timer.section("log_step")
    def log_step(tran, worker):
        episode = episodes[worker]
        episode.add("score", tran["reward"], agg="sum")
        episode.add("length", 1, agg="sum")
        episode.add("rewards", tran["reward"], agg="stack")

        if tran["is_first"]:
            episode.reset()

        if worker < run_args.log_video_streams:
            for key in run_args.log_keys_video:
                if key in tran:
                    episode.add(f"policy_{key}", tran[key], agg="stack")
        for key, value in tran.items():
            if re.match(run_args.log_keys_sum, key):
                episode.add(key, value, agg="sum")
            if re.match(run_args.log_keys_avg, key):
                episode.add(key, value, agg="avg")
            if re.match(run_args.log_keys_max, key):
                episode.add(key, value, agg="max")

        if tran["is_last"]:
            result = episode.result()
            logger.add(
                {
                    "score": result.pop("score"),
                    "length": result.pop("length") - 1,
                },
                prefix="episode",
            )
            rew = result.pop("rewards")
            if len(rew) > 1:
                result["reward_rate"] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

    driver = DriverParallelEnv(
        env,
        run_args.num_envs,
    )
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(log_step)

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(checkpoint_path, keys=["agent"])

    print("Start evaluation")

    # policy = lambda *run_args: agent.policy(*run_args, mode="eval")
    def policy(*run_args):
        return agent.policy(*run_args, mode="eval")

    driver.reset(agent.init_policy)
    while step < run_args.steps:
        driver(policy, steps=10)
        if should_log(step):
            logger.add(agg.result())
            logger.add(epstats.result(), prefix="epstats")
            logger.add(embodied.timer.stats(), prefix="timer")
            logger.add(usage.stats(), prefix="usage")
            logger.add({"fps/policy": policy_fps.result()})
            logger.write()

    logger.close()


### Helper functions ###
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    algorithm_group = parser.add_argument_group(
        "algorithm arguments",
        description="Arguments for algorithm.",
    )
    algorithm_group.add_argument(
        "--model_size",
        type=str,
        default="debug",
        help="Size of the model to train\n(debug, size12m, size25m, size50m, size100m, size200m, size400m)",
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
