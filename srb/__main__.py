#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
# from simforge.core import AssetRegistry, ModelFileFormat
# from simforge.utils import SF_CACHE_DIR, convert_to_snake_case, logging
import argparse
import sys
from importlib.util import find_spec
from os import path
from typing import Literal


def main():
    def impl(
        subcommand: Literal[
            "agent",
            "gui",
            "ls",
        ],
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
def agent_main():
    def impl(
        agent_subcommand: Literal[
            "collect",
            "train",
            "play",
            "rand",
            "zero",
            "teleop",
            "ros",
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

    impl(**vars(parse_cli_args()))


### GUI ###
def launch_gui():
    raise NotImplementedError()


### List ###
def list_registered():
    # if AssetRegistry.n_assets() == 0:
    #     raise ValueError("Cannot list SimForge assets because none are registered")

    if not find_spec("rich"):
        raise ImportError('The "rich" package is required to list SimForge assets')

    # table = Table(title="SimForge Asset Registry")
    # table.add_column("#", justify="right", style="cyan", no_wrap=True)
    # table.add_column("Type", justify="center", style="magenta", no_wrap=True)
    # table.add_column("Package", justify="center", style="green", no_wrap=True)
    # table.add_column("Name", justify="left", style="blue", no_wrap=True)
    # table.add_column("Semantics", justify="left", style="red")
    # table.add_column("Cached", justify="left", style="yellow")
    # i = 0
    # for asset_type, asset_classes in AssetRegistry.items():
    #     cache_dir_for_type = SF_CACHE_DIR.joinpath(str(asset_type))
    #     for j, asset_class in enumerate(asset_classes):
    #         i += 1
    #         asset_name = convert_to_snake_case(asset_class.__name__)
    #         pkg_name = asset_class.__module__.split(".", 1)[0]
    #         asset_cache_dir = cache_dir_for_type.joinpath(asset_name)
    #         asset_cache = {
    #             path.name: len(
    #                 [asset for asset in os.listdir(path) if not asset.endswith(".json")]
    #             )
    #             for path in (
    #                 (
    #                     asset_cache_dir.joinpath(hexdigest)
    #                     for hexdigest in os.listdir(asset_cache_dir)
    #                 )
    #                 if asset_cache_dir.is_dir()
    #                 else ()
    #             )
    #             if path.is_dir()
    #         }
    #         table.add_row(
    #             str(i),
    #             str(asset_type),
    #             f"[link=file://{os.path.dirname(inspect.getabsfile(importlib.import_module(pkg_name)))}]{pkg_name}[/link]",
    #             f"[link=vscode://file/{inspect.getabsfile(asset_class)}:{inspect.getsourcelines(asset_class)[1]}]{asset_name}[/link]",
    #             str(asset_class.SEMANTICS),
    #             (
    #                 ""
    #                 if not asset_cache
    #                 else f"[bold][link=file://{asset_cache_dir}]{sum(asset_cache.values())}[/link]:[/bold] "
    #                 + ", ".join(
    #                     f"[[link=file://{asset_cache_dir.joinpath(hexdigest)}]{n_assets}|{hexdigest[:hash_len]}[/link]]"
    #                     for hexdigest, n_assets in sorted(
    #                         asset_cache.items(),
    #                         key=lambda x: x[1],
    #                         reverse=True,
    #                     )
    #                     if n_assets > 0
    #                 )
    #             ),
    #             end_section=(j + 1) == len(asset_classes),
    #         )
    # print(table)


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
            action=_AutoNamespaceTaskAction,
            default="srb/sample_collection",
            required=True,
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

    _gui_parser = subparsers.add_parser(
        "gui",
        help="gui subcommands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )

    list_parser = subparsers.add_parser(
        "ls",
        help="List registered assets and environments"
        + (' (MISSING: "rich" Python package)' if find_spec("rich") else ""),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    list_parser.add_argument(
        "filter",
        help="List filter",
        nargs="*",
        type=str,
        choices=("all", "assets", "object", "robot", "terrain", "env"),
        default=["all"],
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


class _AutoNamespaceTaskAction(argparse.Action):
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


# from omni.isaac.lab.app import AppLauncher

# def launch_app(args: argparse.Namespace) -> AppLauncher:
#     _autoenable_cameras(args)
#     _autoselect_experience(args)

#     launcher = AppLauncher(launcher_args=args)

#     if args.disable_ui:
#         _disable_ui()

#     return launcher


# def shutdown_app(launcher: AppLauncher):
#     launcher.app.close()


def _autoenable_cameras(args: argparse.Namespace):
    if not args.enable_cameras and (args.video or "visual" in args.task):
        args.enable_cameras = True


def _autoselect_experience(args: argparse.Namespace):
    ## Get relative path to the experience
    project_dir = path.dirname(path.dirname(path.realpath(__file__)))
    experience_dir = path.join(project_dir, "apps")

    ## Select the experience based on args
    experience = "srb"
    if args.headless:
        experience += ".headless"
    if args.enable_cameras:
        experience += ".rendering"
    experience += ".kit"

    ## Set the experience
    args.experience = path.join(experience_dir, experience)


def _disable_ui():
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set("/app/window/hideUi", True)
    settings.set("/app/window/fullscreen", True)


if __name__ == "__main__":
    main()
