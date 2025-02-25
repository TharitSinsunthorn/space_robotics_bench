#!/root/isaac-sim/python.sh

import os
import sys

from omni.isaac.lab.app import AppLauncher

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from _cli_utils import add_default_cli_args, argparse, launch_app, shutdown_app

FRAMEWORK_NAME = "robomimic"
ALGO_CFG_ENTRYPOINT_KEY = f"{FRAMEWORK_NAME}_{{ALGO_NAME}}_cfg"


def main(launcher: AppLauncher, args: argparse.Namespace):
    import json
    import time
    import traceback
    from collections import OrderedDict

    import gymnasium
    import numpy as np
    import psutil
    import robomimic.utils.envutils as EnvUtils
    import robomimic.utils.fileutils as FileUtils
    import robomimic.utils.obsutils as ObsUtils
    import robomimic.utils.torchutils as TorchUtils
    import robomimic.utils.trainutils as TrainUtils
    import torch
    from omni.isaac.kit import SimulationApp
    from robomimic.algo import RolloutPolicy, algo_factory
    from robomimic.config import config_factory
    from robomimic.utils.logutils import DataLogger, PrintLogger
    from torch.utils.data import DataLoader

    import space_robotics_bench  # noqa: F401

    ## Extract simulation app
    _sim_app: SimulationApp = launcher.app

    # load config
    if args.task is not None:
        # obtain the configuration entry point
        cfg_entry_point_key = ALGO_CFG_ENTRYPOINT_KEY.format(ALGO_NAME=args.algo)

        print(f"Loading configuration for task: {args.task}")
        cfg_entry_point_file = gymnasium.spec(args.task).kwargs.pop(cfg_entry_point_key)
        # check if entry point exists
        if cfg_entry_point_file is None:
            raise ValueError(
                f"Could not find configuration for the environment: '{args.task}'."
                f" Please check that the gym registry has the entry point: '{cfg_entry_point_key}'."
            )
        # load config from json file
        with open(cfg_entry_point_file) as f:
            ext_cfg = json.load(f)
            config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        raise ValueError("Please provide a task name through CLI arguments.")

    if args.dataset is not None:
        config.train.data = args.dataset
    else:
        config.train.data = os.path.join(
            f"./logs/{FRAMEWORK_NAME}", args.task, "hdf_dataset.hdf5"
        )

    if args.name is not None:
        config.experiment.name = args.name

    # change location of experiment directory
    config.train.output_dir = os.path.abspath(
        os.path.join(f"./logs/{FRAMEWORK_NAME}", args.task)
    )
    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        # first set seeds
        np.random.seed(config.train.seed)
        torch.manual_seed(config.train.seed)

        print("\n============= New Training Run with Config =============")
        print(config)
        print("")
        log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)
        print(f">>> Saving logs into directory: {log_dir}")
        print(f">>> Saving checkpoints into directory: {ckpt_dir}")
        print(f">>> Saving videos into directory: {video_dir}")

        if config.experiment.logging.terminal_output_to_txt:
            # log stdout and stderr to a text file
            logger = PrintLogger(os.path.join(log_dir, "log.txt"))
            sys.stdout = logger
            sys.stderr = logger

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        ObsUtils.initialize_obsutils_with_config(config)

        # make sure the dataset exists
        dataset_path = os.path.expanduser(config.train.data)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset at provided path {dataset_path} not found!"
            )

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path=config.train.data
        )
        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=config.train.data,
            all_obs_keys=config.all_obs_keys,
            verbose=True,
        )

        if config.experiment.env is not None:
            env_meta["env_name"] = config.experiment.env
            print(
                "=" * 30
                + "\n"
                + "Replacing Env to {}\n".format(env_meta["env_name"])
                + "=" * 30
            )

        # create environment
        envs = OrderedDict()
        if config.experiment.rollout.enabled:
            # create environments for validation runs
            env_names = [env_meta["env_name"]]

            if config.experiment.additional_envs is not None:
                for name in config.experiment.additional_envs:
                    env_names.append(name)

            for env_name in env_names:
                env = EnvUtils.create_env_from_metadata(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"],
                )
                envs[env.name] = env
                print(envs[env.name])

        print("")

        # setup for a new training run
        data_logger = DataLogger(
            log_dir, config=config, log_tb=config.experiment.logging.log_tb
        )
        model = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            device=device,
        )

        # save the config as a json file
        with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
            json.dump(config, outfile, indent=4)

        print("\n============= Model Summary =============")
        print(model)  # print model summary
        print("")

        # load training data
        trainset, validset = TrainUtils.load_data_for_training(
            config, obs_keys=shape_meta["all_obs_keys"]
        )
        train_sampler = trainset.get_dataset_sampler()
        print("\n============= Training Dataset =============")
        print(trainset)
        print("")

        # maybe retrieve statistics for normalizing observations
        obs_normalization_stats = None
        if config.train.hdf5_normalize_obs:
            obs_normalization_stats = trainset.get_obs_normalization_stats()

        # initialize data loaders
        train_loader = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True,
        )

        if config.experiment.validate:
            # cap num workers for validation dataset at 1
            num_workers = min(config.train.num_data_workers, 1)
            valid_sampler = validset.get_dataset_sampler()
            valid_loader = DataLoader(
                dataset=validset,
                sampler=valid_sampler,
                batch_size=config.train.batch_size,
                shuffle=(valid_sampler is None),
                num_workers=num_workers,
                drop_last=True,
            )
        else:
            valid_loader = None

        # main training loop
        best_valid_loss = None
        best_return = (
            {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
        )
        best_success_rate = (
            {k: -1.0 for k in envs} if config.experiment.rollout.enabled else None
        )
        last_ckpt_time = time.time()

        # number of learning steps per epoch (defaults to a full dataset pass)
        train_num_steps = config.experiment.epoch_every_n_steps
        valid_num_steps = config.experiment.validation_epoch_every_n_steps

        for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
            step_log = TrainUtils.run_epoch(
                model=model,
                data_loader=train_loader,
                epoch=epoch,
                num_steps=train_num_steps,
            )
            model.on_epoch_end(epoch)

            # setup checkpoint path
            epoch_ckpt_name = f"model_epoch_{epoch}"

            # check for recurring checkpoint saving conditions
            should_save_ckpt = False
            if config.experiment.save.enabled:
                time_check = (config.experiment.save.every_n_seconds is not None) and (
                    time.time() - last_ckpt_time
                    > config.experiment.save.every_n_seconds
                )
                epoch_check = (
                    (config.experiment.save.every_n_epochs is not None)
                    and (epoch > 0)
                    and (epoch % config.experiment.save.every_n_epochs == 0)
                )
                epoch_list_check = epoch in config.experiment.save.epochs
                should_save_ckpt = time_check or epoch_check or epoch_list_check
            ckpt_reason = None
            if should_save_ckpt:
                last_ckpt_time = time.time()
                ckpt_reason = "time"

            print(f"Train Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Train/{k}", v, epoch)

            # Evaluate the model on validation set
            if config.experiment.validate:
                with torch.no_grad():
                    step_log = TrainUtils.run_epoch(
                        model=model,
                        data_loader=valid_loader,
                        epoch=epoch,
                        validate=True,
                        num_steps=valid_num_steps,
                    )
                for k, v in step_log.items():
                    if k.startswith("Time_"):
                        data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                    else:
                        data_logger.record(f"Valid/{k}", v, epoch)

                print(f"Validation Epoch {epoch}")
                print(json.dumps(step_log, sort_keys=True, indent=4))

                # save checkpoint if achieve new best validation loss
                valid_check = "Loss" in step_log
                if valid_check and (
                    best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)
                ):
                    best_valid_loss = step_log["Loss"]
                    if (
                        config.experiment.save.enabled
                        and config.experiment.save.on_best_validation
                    ):
                        epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                        should_save_ckpt = True
                        ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

            # Evaluate the model by by running rollouts

            # do rollouts at fixed rate or if it's time to save a new ckpt
            video_paths = None
            rollout_check = (epoch % config.experiment.rollout.rate == 0) or (
                should_save_ckpt and ckpt_reason == "time"
            )
            if (
                config.experiment.rollout.enabled
                and (epoch > config.experiment.rollout.warmstart)
                and rollout_check
            ):
                # wrap model as a RolloutPolicy to prepare for rollouts
                rollout_model = RolloutPolicy(
                    model, obs_normalization_stats=obs_normalization_stats
                )

                num_episodes = config.experiment.rollout.n
                all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                    policy=rollout_model,
                    envs=envs,
                    horizon=config.experiment.rollout.horizon,
                    use_goals=config.use_goals,
                    num_episodes=num_episodes,
                    render=False,
                    video_dir=video_dir if config.experiment.render_video else None,
                    epoch=epoch,
                    video_skip=config.experiment.get("video_skip", 5),
                    terminate_on_success=config.experiment.rollout.terminate_on_success,
                )

                # summarize results from rollouts to tensorboard and terminal
                for env_name in all_rollout_logs:
                    rollout_logs = all_rollout_logs[env_name]
                    for k, v in rollout_logs.items():
                        if k.startswith("Time_"):
                            data_logger.record(
                                f"Timing_Stats/Rollout_{env_name}_{k[5:]}", v, epoch
                            )
                        else:
                            data_logger.record(
                                f"Rollout/{k}/{env_name}", v, epoch, log_stats=True
                            )

                    print(
                        "\nEpoch {} Rollouts took {}s (avg) with results:".format(
                            epoch, rollout_logs["time"]
                        )
                    )
                    print(f"Env: {env_name}")
                    print(json.dumps(rollout_logs, sort_keys=True, indent=4))

                # checkpoint and video saving logic
                updated_stats = TrainUtils.should_save_from_rollout_logs(
                    all_rollout_logs=all_rollout_logs,
                    best_return=best_return,
                    best_success_rate=best_success_rate,
                    epoch_ckpt_name=epoch_ckpt_name,
                    save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                    save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
                )
                best_return = updated_stats["best_return"]
                best_success_rate = updated_stats["best_success_rate"]
                epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
                should_save_ckpt = (
                    config.experiment.save.enabled and updated_stats["should_save_ckpt"]
                ) or should_save_ckpt
                if updated_stats["ckpt_reason"] is not None:
                    ckpt_reason = updated_stats["ckpt_reason"]

            # Only keep saved videos if the ckpt should be saved (but not because of validation score)
            should_save_video = (
                should_save_ckpt and (ckpt_reason != "valid")
            ) or config.experiment.keep_all_videos
            if video_paths is not None and not should_save_video:
                for env_name in video_paths:
                    os.remove(video_paths[env_name])

            # Save model checkpoints based on conditions (success rate, validation loss, etc)
            if should_save_ckpt:
                TrainUtils.save_model(
                    model=model,
                    config=config,
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                    obs_normalization_stats=obs_normalization_stats,
                )

            # Finally, log memory usage in MB
            process = psutil.Process(os.getpid())
            mem_usage = int(process.memory_info().rss / 1000000)
            data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
            print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

        # terminate logging
        data_logger.close()

    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)


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
        default="bc",
        help="Name of the algorithm\n(bc, bcq)",
    )
    algorithm_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )
    algorithm_group.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
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
