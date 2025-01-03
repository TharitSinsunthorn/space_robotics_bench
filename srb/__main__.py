#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import sys
from enum import Enum, auto
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable, Literal

from omni.isaac.lab.app import AppLauncher


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
def agent_main(**kwargs):
    def impl(
        agent_subcommand: Literal[
            "collect", "train", "play", "rand", "zero", "teleop", "ros"
        ],
        **kwargs,
    ):
        match agent_subcommand:
            case "collect":
                raise NotImplementedError()
            case "train":
                raise NotImplementedError()
            case "play":
                raise NotImplementedError()
            case "rand":
                raise NotImplementedError()
            case "zero":
                raise NotImplementedError()
            case "teleop":
                raise NotImplementedError()
            case "ros":
                raise NotImplementedError()
            case _:
                raise ValueError(f'Unknown agent subcommand: "{agent_subcommand}"')

    launcher = AppLauncher(launcher_args=kwargs)
    impl(**kwargs)
    launcher.app.close()

    def launch_app(args: argparse.Namespace) -> AppLauncher:
        from omni.isaac.lab.app import AppLauncher

        _autoenable_cameras(args)
        _autoselect_experience(args)

        launcher = AppLauncher(launcher_args=args)

        if args.disable_ui:
            _disable_ui()

        return launcher

    def shutdown_app(launcher: AppLauncher):
        launcher.app.close()

    def _autoenable_cameras(args: argparse.Namespace):
        if not args.enable_cameras and (args.video or "visual" in args.task):
            args.enable_cameras = True

    def _autoselect_experience(args: argparse.Namespace):
        from srb.utils.path import SRB_APPS_DIR

        ## Get relative path to the experience
        ## Select the experience based on args
        experience = "srb"
        if args.headless:
            experience += ".headless"
        if args.enable_cameras:
            experience += ".rendering"
        experience += ".kit"

        ## Set the experience
        args.experience = SRB_APPS_DIR.joinpath(experience).as_posix

    def _disable_ui():
        import carb.settings

        settings = carb.settings.get_settings()
        settings.set("/app/window/hideUi", True)
        settings.set("/app/window/fullscreen", True)


### GUI ###
def launch_gui(release: bool):
    import subprocess

    from srb.utils import logging
    from srb.utils.path import SRB_DIR

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

    from omni.isaac.lab.app import AppLauncher

    # Launch Isaac Sim
    launcher = AppLauncher(headless=True)

    import importlib
    import inspect

    import gymnasium
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
    collect_agent_parser = agent_subparsers.add_parser(
        "collect",
        help="Collect demonstrations",
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
    rand_agent_parser = agent_subparsers.add_parser(
        "rand",
        help="Random agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    zero_agent_parser = agent_subparsers.add_parser(
        "zero",
        help="Zero agent",
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

    for p in (
        collect_agent_parser,
        train_agent_parser,
        play_agent_parser,
        rand_agent_parser,
        zero_agent_parser,
        teleop_agent_parser,
        ros_agent_parser,
    ):
        group = p.add_argument_group("Environment")
        group.add_argument(
            # TODO: Make --env first
            "--task",
            "--env",
            "--demo",
            help="Name of the environment to select",
            type=str,
            action=AutoNamespaceTaskAction,
            default="srb/sample_collection",
        )
        group.add_argument(
            "--seed", type=int, default=None, help="Seed used for the environment"
        )
        group.add_argument(
            "--num_envs",
            type=int,
            default=1,
            help="Number of environments to simulate in parallel.",
        )

        # Compute
        compute_group = p.add_argument_group("Compute")
        compute_group.add_argument(
            "--disable_fabric",
            action="store_true",
            default=False,
            help="Disable fabric and use USD I/O operations.",
        )

        # Video recording
        video_recording_group = p.add_argument_group("Video")
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

        # Experience
        experience_group = p.add_argument_group("Experience")
        experience_group.add_argument(
            "--disable_ui",
            action="store_true",
            default=False,
            help="Disable most of the Isaac Sim UI and set it to fullscreen.",
        )

        AppLauncher.add_app_launcher_args(p)

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

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        unknown_args = (f'"{arg}"' if " " in arg else arg for arg in unknown_args)
        raise ValueError(f'Unknown args encountered: {" ".join(unknown_args)}')

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
