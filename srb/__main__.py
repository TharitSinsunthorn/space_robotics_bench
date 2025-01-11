#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import sys
from enum import Enum, auto
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

import gymnasium
from omni.isaac.lab.app import AppLauncher

from srb.utils.path import SRB_APPS_DIR, SRB_DIR

if TYPE_CHECKING:
    from omni.isaac.kit import SimulationApp

    from srb.core.envs import BaseEnv

# TODO: Clean-up args


def main():
    def impl(
        subcommand: Literal["agent", "gui", "ls"],
        **kwargs,
    ):
        match subcommand:
            case "agent":
                agent_main(**kwargs)
            case "gui":
                launch_gui(**kwargs)
            case "ls":
                list_registered(**kwargs)
            case _:
                raise ValueError(f'Unknown subcommand: "{subcommand}"')

    impl(**vars(parse_cli_args()))


### Agent ###
def agent_main(
    env_id: str,
    video,
    video_length,
    video_interval,
    device,
    disable_fabric,
    disable_ui: bool,
    **kwargs,
):
    # Preprocess kwargs
    kwargs["experience"] = SRB_APPS_DIR.joinpath(
        f'srb.{"headless." if kwargs["headless"] else ""}{"rendering." if kwargs["enable_cameras"] else ""}kit'
    )
    kwargs["enable_cameras"] = (
        kwargs["enable_cameras"] or video or env_id.endswith("_visual")
    )

    # Launch Isaac Sim
    launcher = AppLauncher(launcher_args=kwargs)

    import srb.task as _  # noqa: F401
    from srb.core.teleop_devices import CbKeyboard
    from srb.utils import logging
    from srb.utils.hydra import hydra_task_config
    from srb.utils.isaacsim import hide_ui
    from srb.utils.parsing import create_logdir_path

    if find_spec("rich"):
        from rich import print

    # Post-launch configuration
    if disable_ui:
        hide_ui()

    @hydra_task_config(
        task_name=env_id,
        agent_cfg_entry_point=f'{kwargs["algo"]}_cfg' if kwargs.get("algo") else None,
    )
    def hydra_main(env_cfg: dict | None = None, agent_cfg: dict | None = None):
        # Create the environment and initialize it
        env = gymnasium.make(
            id=env_id, cfg=env_cfg, render_mode="rgb_array" if video else None
        )
        env.reset()

        # Add wrapper for video recording
        if video:
            logdir = Path(create_logdir_path(kwargs["agent_subcommand"], env_id))
            video_kwargs = {
                "video_folder": logdir.joinpath("videos"),
                "step_trigger": lambda step: step % video_interval == 0,
                "video_length": video_length,
                "disable_logger": True,
            }
            logging.info("Recording videos during training.")
            print(video_kwargs)
            env = gymnasium.wrappers.RecordVideo(env, **video_kwargs)

        # Add keyboard callbacks
        if not kwargs["headless"] and kwargs["agent_subcommand"] not in [
            "teleop",
            "collect",
        ]:
            _cb_keyboard = CbKeyboard({"L": env.reset})

        # Run the implementation
        def agent_impl(
            agent_subcommand: Literal[
                "zero",
                "rand",
                "teleop",
                "ros",
                "train",
                "play",
                "collect",
                "learn",
            ],
            **kwargs,
        ):
            match agent_subcommand:
                case "zero":
                    zero_agent(**kwargs)
                case "rand":
                    random_agent(**kwargs)
                case "teleop":
                    teleop_agent(**kwargs)
                case "ros":
                    ros_agent(**kwargs)
                case "train":
                    raise NotImplementedError()
                case "play":
                    raise NotImplementedError()
                case "collect":
                    raise NotImplementedError()
                case "learn":
                    # NOTE: Learning from demonstration does not require the environment
                    raise NotImplementedError()
                case _:
                    raise ValueError(f'Unknown agent subcommand: "{agent_subcommand}"')

        agent_impl(env=env, sim_app=launcher.app, **kwargs)

        # Close the environment
        env.close()

    hydra_main()

    # Shutdown Isaac Sim
    launcher.app.close()


def random_agent(
    env: "BaseEnv",
    sim_app: "SimulationApp",
    **kwargs,
):
    import torch

    # from srb.utils import logging

    with torch.inference_mode():
        while sim_app.is_running():
            actions = torch.from_numpy(env.action_space.sample()).to(device=env.device)

            observation, reward, terminated, truncated, info = env.step(actions)

            # logging.debug(
            #     f"actions: {actions}\n"
            #     f"observation: {observation}\n"
            #     f"reward: {reward}\n"
            #     f"terminated: {terminated}\n"
            #     f"truncated: {truncated}\n"
            #     f"info: {info}\n"
            # )


def zero_agent(
    env: "BaseEnv",
    sim_app: "SimulationApp",
    **kwargs,
):
    import torch

    # from srb.utils import logging

    actions = torch.zeros(env.action_space.shape, device=env.device)  # type: ignore

    with torch.inference_mode():
        while sim_app.is_running():
            observation, reward, terminated, truncated, info = env.step(actions)

            # logging.debug(
            #     f"actions: {actions}\n"
            #     f"observation: {observation}\n"
            #     f"reward: {reward}\n"
            #     f"terminated: {terminated}\n"
            #     f"truncated: {truncated}\n"
            #     f"info: {info}\n"
            # )


def teleop_agent(
    env: "BaseEnv",
    sim_app: "SimulationApp",
    headless: bool,
    teleop_device: Sequence[str],
    pos_sensitivity: float,
    rot_sensitivity: float,
    ros2_integration: bool,
    gui_integration: bool,
    disable_control_scheme_inversion: bool,
    **kwargs,
):
    import threading

    import numpy as np
    import torch
    from rclpy.executors import MultiThreadedExecutor

    from srb.core import mdp
    from srb.core.actions import (
        ManipulatorTaskSpaceActionCfg,
        MultiCopterActionGroupCfg,
        SpacecraftActionGroupCfg,
        WheeledRoverActionGroupCfg,
    )
    from srb.core.interfaces import ROS2, GuiInterface
    from srb.core.managers import SceneEntityCfg
    from srb.core.teleop_devices import CombinedInterface
    from srb.utils.ros import enable_ros2_bridge

    enable_ros2_bridge()
    import rclpy
    from rclpy.node import Node

    if headless and len(teleop_device) == 1 and "keyboard" in teleop_device:
        raise ValueError("Native teleoperation is only supported in GUI mode.")

    # Disable truncation
    if hasattr(env.cfg, "enable_truncation"):
        env.cfg.enable_truncation = False

    # ROS 2 node
    rclpy.init(args=None)
    ros_node = Node("srb")  # type: ignore

    ## Teleop interface
    teleop_interface = CombinedInterface(
        devices=teleop_device,
        node=ros_node,
        pos_sensitivity=pos_sensitivity,
        rot_sensitivity=rot_sensitivity,
        action_cfg=env.cfg.actions,
    )

    def cb_reset():
        global should_reset
        should_reset = True

    global should_reset
    should_reset = False
    teleop_interface.add_callback("L", cb_reset)

    teleop_interface.reset()

    if find_spec("rich"):
        from rich import print
    print(teleop_interface)

    ## ROS 2 interface
    if ros2_integration:
        ros2_interface = ROS2(env, node=ros_node)

    ## GUI interface
    if gui_integration:
        gui_interface = GuiInterface(env, node=ros_node)

    ## Initialize the environment
    observation, info = env.reset()

    def process_actions(
        twist: np.ndarray | torch.Tensor, gripper_cmd: bool
    ) -> torch.Tensor:
        twist = torch.tensor(twist, dtype=torch.float32, device=env.device).repeat(
            env.num_envs, 1
        )
        if isinstance(env.cfg.actions, ManipulatorTaskSpaceActionCfg):
            if not disable_control_scheme_inversion:
                twist[:, :2] *= -1.0
            gripper_action = torch.zeros(twist.shape[0], 1, device=twist.device)
            gripper_action[:] = -1.0 if gripper_cmd else 1.0
            return torch.concat([twist, gripper_action], dim=1)
        elif isinstance(env.cfg.actions, MultiCopterActionGroupCfg):
            return torch.concat(
                [
                    twist[:, :3],
                    twist[:, 5].unsqueeze(1),
                ],
                dim=1,
            )

        elif isinstance(env.cfg.actions, WheeledRoverActionGroupCfg):
            return twist[:, :2]

        elif isinstance(env.cfg.actions, SpacecraftActionGroupCfg):
            return twist[:, :6]
        else:
            raise NotImplementedError()

    # ROS 2 executor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(ros_node)
    thread = threading.Thread(target=executor.spin)
    thread.daemon = True
    thread.start()

    ## Run the environment
    with torch.inference_mode():
        while sim_app.is_running():
            # Get actions from the teleoperation interface
            actions = process_actions(*teleop_interface.advance())

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(actions)

            # Note: Each environment is automatically reset (independently) when terminated or truncated

            # Provide force feedback for teleop devices
            if isinstance(env.cfg.actions, ManipulatorTaskSpaceActionCfg):
                FT_FEEDBACK_SCALE = torch.tensor([0.16, 0.16, 0.16, 0.0, 0.0, 0.0])
                ft_feedback_asset_cfg = SceneEntityCfg(
                    "robot",
                    body_names=env.cfg.robot_cfg.regex_links_hand,
                )
                ft_feedback_asset_cfg.resolve(env.scene)
                ft_feedback = (
                    FT_FEEDBACK_SCALE
                    * mdp.body_incoming_wrench_mean(
                        env=env,
                        asset_cfg=ft_feedback_asset_cfg,
                    )[0, ...].cpu()
                )
                teleop_interface.set_ft_feedback(ft_feedback)

            ## ROS 2 interface
            if ros2_integration:
                ros2_interface.publish(observation, reward, terminated, truncated, info)
                ros2_interface.update()

            ## GUI interface
            if gui_integration:
                gui_interface.update()

            if should_reset:
                should_reset = False
                teleop_interface.reset()
                observation, info = env.reset()


def ros_agent(
    env: "BaseEnv",
    sim_app: "SimulationApp",
    **kwargs,
):
    import torch

    from srb.core.interfaces import ROS2

    # Disable truncation
    if hasattr(env.cfg, "enable_truncation"):
        env.cfg.enable_truncation = False

    ## Create ROS 2 interface
    ros2_interface = ROS2(env)

    ## Run the environment with ROS 2 interface
    with torch.inference_mode():
        while sim_app.is_running():
            # Get actions from ROS 2
            actions = ros2_interface.actions

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(actions)

            # Publish to ROS 2
            ros2_interface.publish(observation, reward, terminated, truncated, info)

            # Process requests from ROS 2
            ros2_interface.update()

            # Note: Each environment is automatically reset (independently) when terminated or truncated


### GUI ###
def launch_gui(release: bool):
    import subprocess

    from srb.utils import logging

    try:
        args = [
            "cargo",
            "run",
            "--manifest-path",
            SRB_DIR.joinpath("Cargo.toml").as_posix(),
            "--package",
            "srb_gui",
            "--bin",
            "gui",
        ] + (["--release"] if release else [])
        logging.info(
            "Launching GUI of the Space Robotics Bench with the following command: "
            + " ".join((f'"{arg}"' if " " in arg else arg) for arg in args)
        )
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        logging.critical("Launching GUI failed due to the exception above")
        exit(e.returncode)


### List ###
def list_registered(category: str | Iterable[str], show_all: bool, **kwargs):
    if not find_spec("rich"):
        raise ImportError(
            'The "rich" package is required to list registered entities of the Space Robotics Bench'
        )

    # Launch Isaac Sim
    launcher = AppLauncher(
        headless=True, experience=SRB_APPS_DIR.joinpath("srb.barebones.kit")
    )

    import importlib
    import inspect

    from rich import print
    from rich.table import Table

    from srb.core.asset import AssetRegistry, AssetType, RobotRegistry
    from srb.utils.registry import get_srb_tasks
    from srb.utils.str import convert_to_snake_case

    # Standardize category
    category = (
        {RegisteredEntity.from_str(category)}
        if isinstance(category, str)
        else set(map(RegisteredEntity.from_str, category))
    )
    if RegisteredEntity.ALL in category:
        category = {RegisteredEntity.ASSET, RegisteredEntity.ENV}
    if RegisteredEntity.ASSET in category:
        category.remove(RegisteredEntity.ASSET)
        category.add(RegisteredEntity.OBJECT)
        category.add(RegisteredEntity.TERRAIN)
        category.add(RegisteredEntity.ROBOT)

    # Print table for assets
    if (
        RegisteredEntity.OBJECT in category
        or RegisteredEntity.TERRAIN in category
        or RegisteredEntity.ROBOT in category
    ):
        table = Table(title="Assets of the Space Robotics Bench")
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Type", justify="center", style="magenta", no_wrap=True)
        table.add_column("Subtype", justify="center", style="red", no_wrap=True)
        table.add_column("Parent Class", justify="left", style="green", no_wrap=True)
        table.add_column("Name", justify="left", style="blue", no_wrap=True)
        i = 0
        if RegisteredEntity.OBJECT in category:
            import srb.asset.object as _  # noqa: F401

            asset_type = AssetType.OBJECT
            asset_classes = AssetRegistry.registry.get(asset_type, ())
            for j, asset_class in enumerate(asset_classes):
                i += 1
                asset_name = convert_to_snake_case(asset_class.__name__)
                parent_class = asset_class.__bases__[0]
                table.add_row(
                    str(i),
                    str(asset_type),
                    "",
                    f"[link=vscode://file/{inspect.getabsfile(parent_class)}:{inspect.getsourcelines(parent_class)[1]}]{parent_class.__name__}[/link]",
                    f"[link=vscode://file/{inspect.getabsfile(asset_class)}:{inspect.getsourcelines(asset_class)[1]}]{asset_name}[/link]",
                    end_section=(j + 1) == len(asset_classes),
                )
        if RegisteredEntity.TERRAIN in category:
            import srb.asset.terrain as _  # noqa: F401

            asset_type = AssetType.TERRAIN
            asset_classes = AssetRegistry.registry.get(asset_type, ())
            for j, asset_class in enumerate(asset_classes):
                i += 1
                asset_name = convert_to_snake_case(asset_class.__name__)
                parent_class = asset_class.__bases__[0]
                table.add_row(
                    str(i),
                    str(asset_type),
                    "",
                    f"[link=vscode://file/{inspect.getabsfile(parent_class)}:{inspect.getsourcelines(parent_class)[1]}]{parent_class.__name__}[/link]",
                    f"[link=vscode://file/{inspect.getabsfile(asset_class)}:{inspect.getsourcelines(asset_class)[1]}]{asset_name}[/link]",
                    end_section=(j + 1) == len(asset_classes),
                )
        if RegisteredEntity.ROBOT in category:
            import srb.asset.robot as _  # noqa: F401

            asset_type = AssetType.ROBOT
            for asset_subtype, asset_classes in RobotRegistry.items():
                for j, asset_class in enumerate(asset_classes):
                    i += 1
                    asset_name = convert_to_snake_case(asset_class.__name__)
                    parent_class = asset_class.__bases__[0]
                    table.add_row(
                        str(i),
                        str(asset_type),
                        str(asset_subtype),
                        f"[link=vscode://file/{inspect.getabsfile(parent_class)}:{inspect.getsourcelines(parent_class)[1]}]{parent_class.__name__}[/link]",
                        f"[link=vscode://file/{inspect.getabsfile(asset_class)}:{inspect.getsourcelines(asset_class)[1]}]{asset_name}[/link]",
                        end_section=(j + 1) == len(asset_classes),
                    )
        print(table)

    # Print table for environments
    if RegisteredEntity.ENV in category:
        import srb.task as srb_tasks

        table = Table(title="Environments of the Space Robotics Bench")
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Type", justify="center", style="bold blue", no_wrap=True)
        table.add_column("ID", justify="left", style="blue", no_wrap=True)
        table.add_column("Entrypoint", justify="left", style="green")
        table.add_column("Config", justify="left", style="yellow")
        table.add_column("Path", justify="left", style="white")
        i = 0
        for task_id in get_srb_tasks():
            if not show_all and task_id.endswith("_visual"):
                continue
            i += 1
            env = gymnasium.registry[task_id]
            entrypoint_str = env.entry_point
            entrypoint_module, entrypoint_class = str(entrypoint_str).split(":")
            env_module_path = inspect.getabsfile(
                importlib.import_module(entrypoint_module.rsplit(".", 1)[0])
            )
            entrypoint_module = importlib.import_module(entrypoint_module)
            entrypoint_class = getattr(entrypoint_module, entrypoint_class)
            entrypoint_parent = entrypoint_class.__bases__[0]
            cfg_class = env.kwargs["task_cfg"]
            cfg_parent = cfg_class.__bases__[0]
            table.add_row(
                str(i),
                "demo" if "demo" in entrypoint_module.__name__ else "task",
                task_id.removeprefix("srb/"),
                f"[link=vscode://file/{inspect.getabsfile(entrypoint_class)}:{inspect.getsourcelines(entrypoint_class)[1]}]{entrypoint_class.__name__}[/link]([red][link=vscode://file/{inspect.getabsfile(entrypoint_parent)}:{inspect.getsourcelines(entrypoint_parent)[1]}]{entrypoint_parent.__name__}[/link][/red])",
                f"[link=vscode://file/{inspect.getabsfile(cfg_class)}:{inspect.getsourcelines(cfg_class)[1]}]{cfg_class.__name__}[/link]([magenta][link=vscode://file/{inspect.getabsfile(cfg_parent)}:{inspect.getsourcelines(cfg_parent)[1]}]{cfg_parent.__name__}[/link][/magenta])",
                f"[link=vscode://file/{env_module_path}]{Path(env_module_path).parent.relative_to(Path(inspect.getabsfile(srb_tasks)).parent)}[/link]",
            )
        print(table)

    # Shutdown Isaac Sim
    launcher.app.close()


class RegisteredEntity(str, Enum):
    ALL = auto()
    ASSET = auto()
    ENV = auto()
    OBJECT = auto()
    ROBOT = auto()
    TERRAIN = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, string: str) -> RegisteredEntity:
        try:
            return next(format for format in cls if string.upper() == format.name)
        except StopIteration:
            raise ValueError(f'String "{string}" is not a valid "{cls.__name__}"')


### CLI ###
def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments for this script.
    """

    parser = argparse.ArgumentParser(
        description="Space Robotics Bench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        required=True,
    )

    # Agent
    agent_parser = subparsers.add_parser(
        "agent",
        help="Agent subcommands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    agent_subparsers = agent_parser.add_subparsers(
        title="Agent subcommands",
        dest="agent_subcommand",
        required=True,
    )
    zero_agent_parser = agent_subparsers.add_parser(
        "zero",
        help="Zero agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    rand_agent_parser = agent_subparsers.add_parser(
        "rand",
        help="Random agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    teleop_agent_parser = agent_subparsers.add_parser(
        "teleop",
        help="Teleoperate agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    ros_agent_parser = agent_subparsers.add_parser(
        "ros",
        help="ROS 2 or Space ROS agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    train_agent_parser = agent_subparsers.add_parser(
        "train",
        help="Train agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    play_agent_parser = agent_subparsers.add_parser(
        "play",
        help="Play agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    collect_agent_parser = agent_subparsers.add_parser(
        "collect",
        help="Collect demonstrations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    learn_agent_parser = agent_subparsers.add_parser(
        "learn",
        help="Learn from demonstrations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )

    for _agent_parser in (
        zero_agent_parser,
        rand_agent_parser,
        teleop_agent_parser,
        ros_agent_parser,
        train_agent_parser,
        play_agent_parser,
        collect_agent_parser,
        learn_agent_parser,
    ):
        environment_group = _agent_parser.add_argument_group("Environment")
        environment_group.add_argument(
            "-e",
            "--env",
            "--task",
            "--demo",
            dest="env_id",
            help="Name of the environment to select",
            type=str,
            action=AutoNamespaceTaskAction,
            default="srb/sample_collection",
        )
        environment_group.add_argument(
            "--seed", type=int, default=None, help="Seed used for the environment"
        )

        compute_group = _agent_parser.add_argument_group("Compute")
        compute_group.add_argument(
            "--disable_fabric",
            action="store_true",
            default=False,
            help="Disable fabric and use USD I/O operations.",
        )

        video_recording_group = _agent_parser.add_argument_group("Video")
        video_recording_group.add_argument(
            "--video",
            action="store_true",
            default=False,
            help="Record videos.",
        )
        video_recording_group.add_argument(
            "--video_length",
            type=int,
            default=1000,
            help="Length of the recorded video (in steps).",
        )
        video_recording_group.add_argument(
            "--video_interval",
            type=int,
            default=10000,
            help="Interval between video recordings (in steps).",
        )

        experience_group = _agent_parser.add_argument_group("Experience")
        experience_group.add_argument(
            "--disable_ui",
            action="store_true",
            default=False,
            help="Disable most of the Isaac Sim UI and set it to fullscreen.",
        )

        AppLauncher.add_app_launcher_args(_agent_parser)

    for _agent_parser in (
        teleop_agent_parser,
        collect_agent_parser,
    ):
        teleop_group = _agent_parser.add_argument_group("Teleop")
        teleop_group.add_argument(
            "--teleop_device",
            type=str,
            nargs="+",
            default=["keyboard"],  # TODO: Convert to enum
            help="Device for interacting with environment",
        )
        teleop_group.add_argument(
            "--pos_sensitivity",
            type=float,
            default=10.0,
            help="Sensitivity factor for translation.",
        )
        teleop_group.add_argument(
            "--rot_sensitivity",
            type=float,
            default=40.0,
            help="Sensitivity factor for rotation.",
        )
        teleop_group.add_argument(
            "--disable_control_scheme_inversion",
            action="store_true",
            default=False,
            help="Flag to disable inverting the control scheme due to view for manipulation-based tasks.",
        )
        teleop_group.add_argument(
            "--ros2_integration",
            action="store_true",
            default=False,  # TODO: Convert to "integrations" list alongside gui (with enum)
            help="Flag to enable ROS 2 interface for subscribing to per-env actions and publishing per-env observations",
        )
        teleop_group.add_argument(
            "--gui_integration",
            action="store_true",
            default=False,
            help="Flag to enable GUI integration",
        )

    for _agent_parser in (
        train_agent_parser,
        play_agent_parser,
    ):
        algorithm_group = _agent_parser.add_argument_group("Algorithm")
        algorithm_group.add_argument(
            "--algo",
            type=str,
            default="ppo",  # TODO: Enum
            help="Name of the algorithm\n(ppo, sac, ppo_lstm)",
        )
        algorithm_group.add_argument(
            "--model_size",
            type=str,
            default="debug",  # TODO: Enum
            help="Size of the model to train\n(debug, size12m, size25m, size50m, size100m, size200m, size400m)",
        )

    train_group = train_agent_parser.add_argument_group("Train")
    train_group.add_argument(
        "--continue_training",
        action="store_true",
        default=False,
        help="Continue training the model from the checkpoint of the last run.",
    )

    # GUI
    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    gui_parser.add_argument(
        "-r",
        "--release",
        action="store_true",
        default=False,
        help="Run GUI in release mode",
    )

    # List
    list_parser = subparsers.add_parser(
        "ls",
        help="List registered assets and environments"
        + (' (MISSING: "rich" Python package)' if find_spec("rich") else ""),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    list_parser.add_argument(
        "category",
        help="Filter of categories to list",
        nargs="*",
        type=str,
        choices=sorted(map(str, RegisteredEntity)),
        default=str(RegisteredEntity.ALL),
    )
    list_parser.add_argument(
        "-a",
        "--show_all",
        action="store_true",
        default=False,
        help='Show all registered entities ("*_visual" environments are hidden by default)',
    )

    if find_spec("argcomplete"):
        import argcomplete

        argcomplete.autocomplete(parser)

    if "--" in sys.argv:
        sys.argv = [sys.argv[0], *sys.argv[(sys.argv.index("--") + 1) :]]
    # TODO: Announce unknown arguments if they start with '-' and do not contain '=' or similar
    args, unknown_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *unknown_args]

    return args


class AutoNamespaceTaskAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ):
        if "/" not in values:
            DEFAULT_TASK_NAMESPACE: str = "srb"
            values = f"{DEFAULT_TASK_NAMESPACE}/{values}"
        setattr(namespace, self.dest, values)


if __name__ == "__main__":
    main()
