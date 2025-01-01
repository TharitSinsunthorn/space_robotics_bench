#!/root/isaac-sim/python.sh

import os
import sys

from omni.isaac.lab.app import AppLauncher

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from _cli_utils import add_default_cli_args, argparse, launch_app, shutdown_app

FRAMEWORK_NAME = "robomimic"


def main(launcher: AppLauncher, args: argparse.Namespace):
    ## Note: Importing modules here due to delayed Omniverse Kit extension loading
    import contextlib
    from os import path

    import gymnasium
    import numpy as np
    import torch
    from omni.isaac.kit import SimulationApp
    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

    import srb  # Noqa: F401
    from srb.core import mdp
    from srb.core.actions import (
        ManipulatorTaskSpaceActionCfg,
        MultiCopterActionGroupCfg,
        WheeledRoverActionGroupCfg,
    )
    from srb.core.managers import SceneEntityCfg
    from srb.core.teleop_devices import CombinedInterface
    from srb.core.wrappers.robomimic import RobomimicDataCollector
    from srb.utils.parsing import create_logdir_path, parse_task_cfg

    if args.headless and "keyboard" in args.teleop_device:
        raise ValueError("Native teleoperation is only supported in GUI mode.")

    ## Extract simulation app
    sim_app: SimulationApp = launcher.app

    # Parse configuration
    task_cfg = parse_task_cfg(
        task_name=args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    # Disable truncation
    if hasattr(task_cfg, "enable_truncation"):
        task_cfg.enable_truncation = False

    ## Create the environment
    env = gymnasium.make(
        id=args.task, cfg=task_cfg, render_mode="rgb_array" if args.video else None
    )

    ## Create controller
    teleop_interface = CombinedInterface(
        devices=args.teleop_device,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
        action_cfg=env.unwrapped.cfg.actions,
    )
    teleop_interface.reset()
    teleop_interface.add_callback("L", env.reset)
    print(teleop_interface)

    ## Initialize the environment
    observation, info = env.reset()

    ## Add wrapper for video recording (if enabled)
    if args.video:
        logdir = create_logdir_path("srb", args.task)
        video_kwargs = {
            "video_folder": path.join(logdir, "videos"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gymnasium.wrappers.RecordVideo(env, **video_kwargs)

    def process_actions(twist: np.ndarray, gripper_cmd: bool) -> torch.Tensor:
        twist = torch.tensor(
            twist, dtype=torch.float32, device=env.unwrapped.device
        ).repeat(env.unwrapped.num_envs, 1)
        if isinstance(env.unwrapped.cfg.actions, ManipulatorTaskSpaceActionCfg):
            if not args.disable_control_scheme_inversion:
                twist[:, :2] *= -1.0
            gripper_action = torch.zeros(twist.shape[0], 1, device=twist.device)
            gripper_action[:] = -1.0 if gripper_cmd else 1.0
            return torch.concat([twist, gripper_action], dim=1)
        elif isinstance(env.unwrapped.cfg.actions, MultiCopterActionGroupCfg):
            return torch.concat(
                [
                    twist[:, :3],
                    twist[:, 5].unsqueeze(1),
                ],
                dim=1,
            )

        elif isinstance(env.unwrapped.cfg.actions, WheeledRoverActionGroupCfg):
            return twist[:, :2]

    # Specify directory for logging experiments
    logdir = create_logdir_path(FRAMEWORK_NAME, f"{args.task}/dataset/")

    # Dump the configuration into log-directory
    dump_yaml(os.path.join(logdir, "params", "task.yaml"), task_cfg)
    dump_pickle(os.path.join(logdir, "params", "task.pkl"), task_cfg)

    # Create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args.task,
        directory_path=logdir,
        filename=args.filename,
        num_demos=args.num_demos,
        flush_freq=env.num_envs,
    )
    collector_interface.reset()

    ## Run the environment
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while sim_app.is_running() and not collector_interface.is_stopped():
            # Get actions from the teleoperation interface
            actions = process_actions(*teleop_interface.advance())

            # Store actions and observations before stepping the environment
            collector_interface.add("actions", actions)
            for key, value in observation.items():
                if key == "policy":
                    for inner_key, inner_value in value.items():
                        collector_interface.add(f"obs/{inner_key}", inner_value)
                else:
                    collector_interface.add(f"obs/{key}", value)

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            # Note: Each environment is automatically reset (independently) when terminated or truncated

            # Store observations, rewards and dones after stepping the environment
            for key, value in observation.items():
                if key == "policy":
                    for inner_key, inner_value in value.items():
                        collector_interface.add(f"next_obs/{inner_key}", inner_value)
                else:
                    collector_interface.add(f"next_obs/{key}", value)
            collector_interface.add("rewards", reward)
            collector_interface.add("dones", dones)
            collector_interface.add("success", torch.zeros_like(dones))

            # Flush data from collector for successful environments
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(reset_env_ids)

            # Provide force feedback for teleop devices
            if isinstance(env.unwrapped.cfg.actions, ManipulatorTaskSpaceActionCfg):
                FT_FEEDBACK_SCALE = torch.tensor([0.16, 0.16, 0.16, 0.0, 0.0, 0.0])
                ft_feedback_asset_cfg = SceneEntityCfg(
                    "robot",
                    body_names=task_cfg.robot_cfg.regex_links_hand,
                )
                ft_feedback_asset_cfg.resolve(env.unwrapped.scene)
                ft_feedback = (
                    FT_FEEDBACK_SCALE
                    * mdp.body_incoming_wrench_mean(
                        env=env.unwrapped,
                        asset_cfg=ft_feedback_asset_cfg,
                    )[0, ...].cpu()
                )
                teleop_interface.set_ft_feedback(ft_feedback)

            # Check if the data collection is stopped
            if collector_interface.is_stopped():
                break

    # Close the simulator
    collector_interface.close()
    env.close()


### Helper functions ###
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    algorithm_group = parser.add_argument_group(
        "teleop",
        description="Arguments for teleoperation",
    )
    algorithm_group.add_argument(
        "--teleop_device",
        type=str,
        nargs="+",
        default=["keyboard"],
        help="Device for interacting with environment",
    )
    algorithm_group.add_argument(
        "--pos_sensitivity",
        type=float,
        default=10.0,
        help="Sensitivity factor for translation.",
    )
    algorithm_group.add_argument(
        "--rot_sensitivity",
        type=float,
        default=40.0,
        help="Sensitivity factor for rotation.",
    )
    algorithm_group.add_argument(
        "--disable_control_scheme_inversion",
        action="store_true",
        default=False,
        help="Flag to disable inverting the control scheme due to view for manipulation-based tasks.",
    )
    algorithm_group = parser.add_argument_group(
        "data_collection",
        description="Arguments for data collection",
    )
    algorithm_group.add_argument(
        "--num_demos",
        type=int,
        default=1000,
        help="Number of episodes to store in the dataset.",
    )
    algorithm_group.add_argument(
        "--filename", type=str, default="hdf_dataset", help="Basename of output file."
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
