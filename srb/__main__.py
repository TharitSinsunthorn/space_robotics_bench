#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import os
import sys
from enum import Enum, auto
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence, Tuple

if TYPE_CHECKING:
    from omni.isaac.kit import SimulationApp

    from srb._typing import AnyEnv
    from srb.interfaces.gui import GuiInterface
    from srb.interfaces.ros2 import ROS2Interface
    from srb.interfaces.teleop import CombinedTeleopInterface

# TODO: Clean-up args (argparse and functions) with enums, add choices, ...


def main():
    def impl(
        subcommand: Literal["agent", "gui", "ls"],
        **kwargs,
    ):
        if not find_spec("omni"):
            raise ImportError(
                "The Space Robotics Bench requires an environment with NVIDIA Omniverse and Isaac Sim installed."
            )

        match subcommand:
            case "agent":
                agent_main(**kwargs)
            case "gui":
                launch_gui(**kwargs)
            case "ls":
                list_registered(**kwargs)

    impl(**vars(parse_cli_args()))


### Agent ###
def agent_main(
    agent_subcommand: Literal[
        "zero",
        "rand",
        "teleop",
        "ros",
        "train",
        "eval",
        "collect",
        "learn",
    ],
    env_id: str,
    video,
    video_length,
    video_interval,
    hide_ui: bool,
    **kwargs,
):
    from srb.core.app import AppLauncher
    from srb.utils.path import SRB_APPS_DIR

    # Preprocess kwargs
    kwargs["enable_cameras"] = video or env_id.endswith("_visual")
    kwargs["experience"] = SRB_APPS_DIR.joinpath(
        f'srb.{"headless." if kwargs["headless"] else ""}{"rendering." if video or kwargs["enable_cameras"] else ""}kit'
    )

    # Launch Isaac Sim
    launcher = AppLauncher(launcher_args=kwargs)

    from srb import tasks as _  # noqa: F401
    from srb.interfaces.teleop import EventKeyboardTeleopInterface
    from srb.utils import logging
    from srb.utils.cfg import hydra_task_config, last_logdir, new_logdir
    from srb.utils.isaacsim import hide_isaacsim_ui

    # Post-launch configuration
    if hide_ui:
        hide_isaacsim_ui()

    # Get the log directory based on the workflow
    workflow = kwargs.get("algo") or agent_subcommand
    if model := kwargs.get("model"):
        model = Path(model)
        assert model.exists(), f"Model path does not exist: {model}"
        logdir = model.parent
        while not (
            logdir.parent.name == workflow
            and (logdir.parent.parent.name == env_id.removeprefix("srb/"))
        ):
            _new_parent = logdir.parent
            if logdir == _new_parent:
                # TODO: Maybe the dir needs to be created here before symlink
                model_symlink_path = logdir.joinpath("model").joinpath(model.name)
                os.symlink(model, model_symlink_path)
                model = model_symlink_path
                logdir = new_logdir(env_id=env_id, workflow=workflow)
                break
            logdir = _new_parent
        kwargs["model"] = model
    elif (agent_subcommand == "train" and kwargs["continue_training"]) or (
        agent_subcommand in ("eval", "teleop") and kwargs["algo"]
    ):
        logdir = last_logdir(env_id=env_id, workflow=workflow)
    else:
        logdir = new_logdir(env_id=env_id, workflow=workflow)

    @hydra_task_config(
        task_name=env_id,
        agent_cfg_entry_point=f'{kwargs["algo"]}_cfg' if kwargs.get("algo") else None,
    )
    def hydra_main(env_cfg: dict | None = None, agent_cfg: dict | None = None):
        import gymnasium

        # Create the environment and initialize it
        env = gymnasium.make(
            id=env_id, cfg=env_cfg, render_mode="rgb_array" if video else None
        )
        env.reset()

        # Add wrapper for video recording
        if video:
            video_kwargs = {
                "video_folder": logdir.joinpath("videos"),
                "step_trigger": lambda step: step % video_interval == 0,
                "video_length": video_length,
                "disable_logger": True,
            }
            logging.info(f"Recording videos: {video_kwargs}")
            env = gymnasium.wrappers.RecordVideo(env, **video_kwargs)

        # Add keyboard callbacks
        if not kwargs["headless"] and agent_subcommand not in [
            "teleop",
            "collect",
            "train",
        ]:
            _cb_keyboard = EventKeyboardTeleopInterface({"L": env.reset})

        # Run the implementation
        def agent_impl(**kwargs):
            kwargs.update(
                {
                    "env_id": env_id,
                    "agent_cfg": agent_cfg,
                    "env_cfg": env_cfg,
                }
            )

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
                    train_agent(**kwargs)
                case "eval":
                    eval_agent(**kwargs)
                case "collect":
                    raise NotImplementedError()
                case "learn":
                    # NOTE: Learning from demonstration does not require the environment
                    raise NotImplementedError()

        agent_impl(env=env, sim_app=launcher.app, logdir=logdir, **kwargs)

        # Close the environment
        env.close()

    hydra_main()

    # Shutdown Isaac Sim
    launcher.app.close()


def random_agent(
    env: "AnyEnv",
    sim_app: "SimulationApp",
    **kwargs,
):
    import torch

    from srb.utils import logging

    with torch.inference_mode():
        while sim_app.is_running():
            actions = torch.from_numpy(env.action_space.sample()).to(device=env.device)

            observation, reward, terminated, truncated, info = env.step(actions)  # type: ignore

            logging.trace(
                f"actions: {actions}\n"
                f"observation: {observation}\n"
                f"reward: {reward}\n"
                f"terminated: {terminated}\n"
                f"truncated: {truncated}\n"
                f"info: {info}\n"
            )


def zero_agent(
    env: "AnyEnv",
    sim_app: "SimulationApp",
    **kwargs,
):
    import torch

    from srb.utils import logging

    actions = torch.zeros(env.action_space.shape, device=env.device)  # type: ignore

    with torch.inference_mode():
        while sim_app.is_running():
            observation, reward, terminated, truncated, info = env.step(actions)

            logging.trace(
                f"actions: {actions}\n"
                f"observation: {observation}\n"
                f"reward: {reward}\n"
                f"terminated: {terminated}\n"
                f"truncated: {truncated}\n"
                f"info: {info}\n"
            )


def teleop_agent(
    env: "AnyEnv",
    sim_app: "SimulationApp",
    headless: bool,
    teleop_device: Sequence[str],
    pos_sensitivity: float,
    rot_sensitivity: float,
    integration: Sequence[str],
    algo: str,
    **kwargs,
):
    from srb.utils.ros2 import enable_ros2_bridge

    enable_ros2_bridge()

    import threading

    import gymnasium

    from srb.core.action import ActionGroup
    from srb.interfaces.teleop import CombinedTeleopInterface

    # Ensure that a feasible teleoperation device is selected
    if headless and len(teleop_device) == 1 and "keyboard" in teleop_device:
        raise ValueError(
            'Teleoperation with the keyboard is only supported in GUI mode. Consider disabling the "--headless" mode or using a different "--teleop_device".'
        )

    # Disable truncation
    if hasattr(env.cfg, "enable_truncation"):
        env.cfg.enable_truncation = False

    # Create ROS 2 node
    if "gui" in integration or "ros2" in integration:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node

        rclpy.init(args=None)
        ros_node = Node("srb")  # type: ignore
    else:
        ros_node = None

    ## Create teleop interface
    teleop_interface = CombinedTeleopInterface(
        devices=teleop_device,
        node=ros_node,
        pos_sensitivity=pos_sensitivity,
        rot_sensitivity=rot_sensitivity,
        action_cfg=env.cfg.robot.action_cfg,  # type: ignore
    )

    ## Set up reset callback
    def cb_reset():
        global should_reset
        should_reset = True

    global should_reset
    should_reset = False
    teleop_interface.add_callback("L", cb_reset)

    ## Initialize the teleop interface via reset
    teleop_interface.reset()
    print(teleop_interface)

    ## Create GUI interface
    if "gui" in integration:
        from srb.interfaces.gui import GuiInterface

        gui_interface = GuiInterface(env, node=ros_node)
    else:
        gui_interface = None

    ## Create ROS 2 interface
    if "ros2" in integration:
        from srb.interfaces.ros2 import ROS2Interface

        ros2_interface = ROS2Interface(env, node=ros_node)
    else:
        ros2_interface = None

    ## Initialize the environment
    env.reset()

    ## Spin up ROS 2 executor
    if ros_node:
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(ros_node)
        thread = threading.Thread(target=executor.spin)
        thread.daemon = True
        thread.start()

    ## Determine how to teleoperate the agent and dispatch the appropriate implementation
    env_supports_direct_teleop = (
        hasattr(env.cfg.robot.action_cfg.__class__, "map_commands")  # type: ignore
        and env.cfg.robot.action_cfg.__class__.map_commands  # type: ignore
        is not ActionGroup.map_commands
    )

    if env_supports_direct_teleop:
        _teleop_agent_direct(
            env=env,
            sim_app=sim_app,
            teleop_interface=teleop_interface,
            ros2_interface=ros2_interface,
            gui_interface=gui_interface,
            **kwargs,
        )
    elif (
        isinstance(env.observation_space, gymnasium.spaces.Dict)
        and "command" in env.observation_space.spaces.keys()
    ):
        if algo:
            _teleop_agent_via_policy(
                env=env,
                sim_app=sim_app,
                teleop_interface=teleop_interface,
                ros2_interface=ros2_interface,
                gui_interface=gui_interface,
                algo=algo,
                **kwargs,
            )
        else:
            raise ValueError(
                f'Environment "{env}" can only be teleoperated via policy. Please provide a policy via "--algo" and an optional "--model" argument.'
            )
    else:
        raise ValueError(f'Environment "{env}" does not support teleoperation.')


def _teleop_agent_direct(
    env: "AnyEnv",
    sim_app: "SimulationApp",
    teleop_interface: "CombinedTeleopInterface",
    ros2_interface: "ROS2Interface | None",
    gui_interface: "GuiInterface | None",
    disable_control_scheme_inversion: bool,
    **kwargs,
):
    import torch

    from srb.core.asset import Manipulator
    from srb.core.manager import SceneEntityCfg
    from srb.core.mdp import body_incoming_wrench_mean

    is_manip_task = isinstance(env.cfg.robot, Manipulator)

    ## Run the environment
    with torch.inference_mode():
        while sim_app.is_running():
            ## Get actions from the teleoperation interface and process them
            twist, event = teleop_interface.advance()
            if is_manip_task and not disable_control_scheme_inversion:
                twist[:2] *= -1.0
            actions = env.cfg.robot.action_cfg.map_commands(  # type: ignore
                torch.from_numpy(twist).to(device=env.device, dtype=torch.float32),
                event,
            ).repeat(env.num_envs, 1)

            ## Step the environment
            observation, reward, terminated, truncated, info = env.step(actions)

            ## Provide force feedback
            # TODO: Generalize force feedback for all tasks
            if is_manip_task:
                FT_FEEDBACK_SCALE = torch.tensor([0.16, 0.16, 0.16, 0.0, 0.0, 0.0])
                ft_feedback_asset_cfg = SceneEntityCfg(
                    "robot",
                    body_names=env.cfg.robot.regex_links_hand,  # type: ignore
                )
                ft_feedback_asset_cfg.resolve(env.scene)
                ft_feedback = (
                    FT_FEEDBACK_SCALE
                    * body_incoming_wrench_mean(
                        env=env,
                        asset_cfg=ft_feedback_asset_cfg,
                    )[0, ...].cpu()
                )
                teleop_interface.set_ft_feedback(ft_feedback)  # type: ignore

            ## Update GUI interface
            if gui_interface:
                gui_interface.update()

            ## Update ROS 2 interface
            if ros2_interface:
                ros2_interface.publish(
                    observation,  # type: ignore
                    reward,  # type: ignore
                    terminated,  # type: ignore
                    truncated,  # type: ignore
                    info,
                )
                ros2_interface.update()

            ## Process reset request
            global should_reset
            if should_reset:
                should_reset = False
                teleop_interface.reset()
                observation, info = env.reset()


def _teleop_agent_via_policy(
    env: "AnyEnv",
    sim_app: "SimulationApp",
    teleop_interface: "CombinedTeleopInterface",
    ros2_interface: "ROS2Interface | None",
    gui_interface: "GuiInterface | None",
    disable_control_scheme_inversion: bool,
    **kwargs,
):
    import torch
    from gymnasium.core import (
        ActType,
        ObservationWrapper,
        ObsType,
        SupportsFloat,
        WrapperObsType,
    )

    from srb.core.asset import Manipulator

    # Disable command randomization
    if hasattr(env.cfg.events, "command"):
        env.cfg.events.command = None  # type: ignore

    is_manip_task = isinstance(env.cfg.robot, Manipulator)

    class InjectTeleopWrapper(ObservationWrapper):
        def observation(self, observation: ObsType) -> WrapperObsType:  # type: ignore
            ## Get actions from the teleoperation interface and process them
            twist, event = teleop_interface.advance()
            if is_manip_task and not disable_control_scheme_inversion:
                twist[:2] *= -1.0

            cmd_len = observation["command"].shape[-1]  # type: ignore

            ## Map teleoperation actions to commands
            match cmd_len:
                case _ if cmd_len < 7:
                    observation["command"][:] = torch.from_numpy(twist[:cmd_len]).to(  # type: ignore
                        device=env.device, dtype=torch.float32
                    )
                case 7:
                    observation["command"][:] = torch.concat(  # type: ignore
                        (
                            torch.from_numpy(twist).to(
                                device=env.device, dtype=torch.float32
                            ),
                            torch.Tensor((-1.0 if event else 1.0,)).to(
                                device=twist.device  # type: ignore
                            ),
                        )
                    )
                case _:
                    raise ValueError(
                        f"Unsupported command length for teleoperation: {cmd_len}"
                    )

            return observation  # type: ignore

        def step(
            self,
            action: ActType,  # type: ignore
        ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Mapping[str, Any]]:  # type: ignore
            # Exit if the simulation is not running
            if not sim_app.is_running():
                exit()

            observation, reward, terminated, truncated, info = super().step(action)

            ## Update GUI interface
            if gui_interface:
                gui_interface.update()

            ## Update ROS 2 interface
            if ros2_interface:
                ros2_interface.publish(
                    observation,  # type: ignore
                    reward,  # type: ignore
                    terminated,  # type: ignore
                    truncated,  # type: ignore
                    info,
                )
                ros2_interface.update()

            ## Process reset request
            global should_reset
            if should_reset:
                should_reset = False
                teleop_interface.reset()
                observation, info = env.reset()

            return (
                observation,  # type: ignore
                reward,
                terminated,
                truncated,
                info,
            )

    # Wrap the environment with the teleoperation interface
    env = InjectTeleopWrapper(env)  # type: ignore

    ## Evaluate the agent with the wrapped environment
    eval_agent(env=env, sim_app=sim_app, **kwargs)


def ros_agent(
    env: "AnyEnv",
    sim_app: "SimulationApp",
    **kwargs,
):
    import torch

    from srb.interfaces.ros2 import ROS2Interface

    # Disable truncation
    if hasattr(env.cfg, "enable_truncation"):
        env.cfg.enable_truncation = False

    ## Create ROS 2 interface
    ros2_interface = ROS2Interface(env)

    ## Run the environment with ROS 2 interface
    with torch.inference_mode():
        while sim_app.is_running():
            # Get actions from ROS 2
            actions = ros2_interface.actions

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(actions)  # type: ignore

            # Publish to ROS 2
            ros2_interface.publish(
                observation,  # type: ignore
                reward,  # type: ignore
                terminated,  # type: ignore
                truncated,  # type: ignore
                info,
            )

            # Process requests from ROS 2
            ros2_interface.update()


def train_agent(
    algo: str,
    **kwargs,
):
    WORKFLOW: str = "train"

    match algo:
        case "dreamer":
            from srb.integrations.dreamer import main as dreamer

            dreamer.run(workflow=WORKFLOW, **kwargs)
        case _skrl if algo.startswith("skrl"):
            from srb.integrations.skrl import main as skrl

            skrl.run(workflow=WORKFLOW, **kwargs)
        case _sb3 if algo.startswith("sb3"):
            from srb.integrations.sb3 import main as sb3

            sb3.run(workflow=WORKFLOW, algo=algo.strip("sb3_"), **kwargs)
        case _sbx if algo.startswith("sbx"):
            from srb.integrations.sbx import main as sbx

            sbx.run(workflow=WORKFLOW, algo=algo.strip("sbx_"), **kwargs)


def eval_agent(
    algo: str,
    **kwargs,
):
    WORKFLOW: str = "eval"

    match algo:
        case "dreamer":
            from srb.integrations.dreamer import main as dreamer

            dreamer.run(workflow=WORKFLOW, **kwargs)
        case _skrl if algo.startswith("skrl"):
            from srb.integrations.skrl import main as skrl

            skrl.run(workflow=WORKFLOW, **kwargs)
        case _sb3 if algo.startswith("sb3"):
            from srb.integrations.sb3 import main as sb3

            sb3.run(workflow=WORKFLOW, algo=algo.strip("sb3_"), **kwargs)
        case _sbx if algo.startswith("sbx"):
            from srb.integrations.sbx import main as sbx

            sbx.run(workflow=WORKFLOW, algo=algo.strip("sbx_"), **kwargs)


### GUI ###
def launch_gui(release: bool):
    import subprocess

    from srb.utils import logging
    from srb.utils.path import SRB_DIR

    try:
        import string

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
            + " ".join(
                (f'"{arg}"' if any(c in string.whitespace for c in arg) else arg)
                for arg in args
            )
        )
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        logging.critical("Launching GUI failed due to the exception above")
        exit(e.returncode)


### List ###
def list_registered(category: str | Iterable[str], show_all: bool, **kwargs):
    from srb.core.app import AppLauncher
    from srb.utils.path import SRB_APPS_DIR

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
    from os import path

    from rich import print
    from rich.table import Table

    from srb.utils.str import convert_to_snake_case

    # Standardize category
    category = (
        {RegisteredEntity.from_str(category)}
        if isinstance(category, str)
        else set(map(RegisteredEntity.from_str, category))
    )
    if RegisteredEntity.ALL in category:
        category = {
            RegisteredEntity.ACTION,
            RegisteredEntity.ASSET,
            RegisteredEntity.ENV,
        }
    if RegisteredEntity.ASSET in category:
        category.remove(RegisteredEntity.ASSET)
        category.add(RegisteredEntity.OBJECT)
        category.add(RegisteredEntity.TERRAIN)
        category.add(RegisteredEntity.ROBOT)

    if RegisteredEntity.ENV in category:
        from srb import tasks as srb_tasks

    # Print table for assets
    if (
        RegisteredEntity.OBJECT in category
        or RegisteredEntity.TERRAIN in category
        or RegisteredEntity.ROBOT in category
    ):
        from srb.core.asset import AssetRegistry, AssetType

        table = Table(title="Assets of the Space Robotics Bench")
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Type", justify="center", style="magenta", no_wrap=True)
        table.add_column("Subtype", justify="center", style="red", no_wrap=True)
        table.add_column("Parent Class", justify="left", style="green", no_wrap=True)
        table.add_column("Asset Cfg", justify="left", style="yellow")
        table.add_column("Name", justify="left", style="blue", no_wrap=True)
        table.add_column("Path", justify="left", style="white")
        i = 0
        if RegisteredEntity.OBJECT in category:
            from srb.assets import object as srb_objects

            asset_type = AssetType.OBJECT
            asset_classes = AssetRegistry.registry.get(asset_type, ())
            for j, asset_class in enumerate(asset_classes):
                i += 1
                asset_name = convert_to_snake_case(asset_class.__name__)
                parent_class = asset_class.__bases__[0]
                try:
                    asset_cfg_class = asset_class().asset_cfg.__class__  # type: ignore
                except Exception:
                    asset_cfg_class = None
                asset_module_path = Path(
                    inspect.getabsfile(importlib.import_module(asset_class.__module__))
                )
                try:
                    asset_module_relpath = asset_module_path.relative_to(
                        Path(inspect.getabsfile(srb_objects)).parent
                    )
                except ValueError:
                    asset_module_relpath = path.join("EXT", asset_module_path.name)
                table.add_row(
                    str(i),
                    str(asset_type),
                    "",
                    f"[link=vscode://file/{inspect.getabsfile(parent_class)}:{inspect.getsourcelines(parent_class)[1]}]{parent_class.__name__}[/link]",
                    f"[link=vscode://file/{inspect.getabsfile(asset_cfg_class)}:{inspect.getsourcelines(asset_cfg_class)[1]}]{asset_cfg_class.__name__}[/link]"
                    if asset_cfg_class
                    else "",
                    f"[link=vscode://file/{inspect.getabsfile(asset_class)}:{inspect.getsourcelines(asset_class)[1]}]{asset_name}[/link]",
                    f"[link=vscode://file/{asset_module_path}]{asset_module_relpath}[/link]",
                    end_section=(j + 1) == len(asset_classes),
                )
        if RegisteredEntity.TERRAIN in category:
            from srb.assets import terrain as srb_terrains

            asset_type = AssetType.TERRAIN
            asset_classes = AssetRegistry.registry.get(asset_type, ())
            for j, asset_class in enumerate(asset_classes):
                i += 1
                asset_name = convert_to_snake_case(asset_class.__name__)
                parent_class = asset_class.__bases__[0]
                try:
                    asset_cfg_class = asset_class().asset_cfg.__class__  # type: ignore
                except Exception:
                    asset_cfg_class = None
                asset_module_path = Path(
                    inspect.getabsfile(importlib.import_module(asset_class.__module__))
                )
                try:
                    asset_module_relpath = asset_module_path.relative_to(
                        Path(inspect.getabsfile(srb_terrains)).parent
                    )
                except ValueError:
                    asset_module_relpath = path.join("EXT", asset_module_path.name)
                table.add_row(
                    str(i),
                    str(asset_type),
                    "",
                    f"[link=vscode://file/{inspect.getabsfile(parent_class)}:{inspect.getsourcelines(parent_class)[1]}]{parent_class.__name__}[/link]",
                    f"[link=vscode://file/{inspect.getabsfile(asset_cfg_class)}:{inspect.getsourcelines(asset_cfg_class)[1]}]{asset_cfg_class.__name__}[/link]"
                    if asset_cfg_class
                    else "",
                    f"[link=vscode://file/{inspect.getabsfile(asset_class)}:{inspect.getsourcelines(asset_class)[1]}]{asset_name}[/link]",
                    f"[link=vscode://file/{asset_module_path}]{asset_module_relpath}[/link]",
                    end_section=(j + 1) == len(asset_classes),
                )
        if RegisteredEntity.ROBOT in category:
            from srb.assets import robot as srb_robots
            from srb.core.asset import RobotRegistry

            asset_type = AssetType.ROBOT
            for asset_subtype, asset_classes in RobotRegistry.items():
                for j, asset_class in enumerate(asset_classes):
                    i += 1
                    asset_name = convert_to_snake_case(asset_class.__name__)
                    parent_class = asset_class.__bases__[0]
                    try:
                        asset_cfg_class = asset_class().asset_cfg.__class__  # type: ignore
                    except Exception:
                        asset_cfg_class = None
                    asset_module_path = Path(
                        inspect.getabsfile(
                            importlib.import_module(asset_class.__module__)
                        )
                    )
                    try:
                        asset_module_relpath = asset_module_path.relative_to(
                            Path(inspect.getabsfile(srb_robots)).parent
                        )
                    except ValueError:
                        asset_module_relpath = path.join("EXT", asset_module_path.name)
                    table.add_row(
                        str(i),
                        str(asset_type),
                        str(asset_subtype),
                        f"[link=vscode://file/{inspect.getabsfile(parent_class)}:{inspect.getsourcelines(parent_class)[1]}]{parent_class.__name__}[/link]",
                        f"[link=vscode://file/{inspect.getabsfile(asset_cfg_class)}:{inspect.getsourcelines(asset_cfg_class)[1]}]{asset_cfg_class.__name__}[/link]"
                        if asset_cfg_class
                        else "",
                        f"[link=vscode://file/{inspect.getabsfile(asset_class)}:{inspect.getsourcelines(asset_class)[1]}]{asset_name}[/link]",
                        f"[link=vscode://file/{asset_module_path}]{asset_module_relpath}[/link]",
                        end_section=(j + 1) == len(asset_classes),
                    )
        print(table)

    # Print table for action groups
    if RegisteredEntity.ACTION in category:
        from srb.core.action import ActionGroupRegistry
        from srb.core.action import group as srb_action_groups

        table = Table(title="Action Groups of the Space Robotics Bench")
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Name", justify="left", style="blue", no_wrap=True)
        table.add_column("Path", justify="left", style="white")

        for i, action_group_class in enumerate(ActionGroupRegistry.registry, 1):
            action_group_name = convert_to_snake_case(action_group_class.__name__)
            action_group_path = Path(
                inspect.getabsfile(
                    importlib.import_module(action_group_class.__module__)
                )
            )
            try:
                action_group_relpath = action_group_path.relative_to(
                    Path(inspect.getabsfile(srb_action_groups)).parent
                )
            except ValueError:
                action_group_relpath = path.join("EXT", action_group_path.name)
            table.add_row(
                str(i),
                f"[link=vscode://file/{inspect.getabsfile(action_group_class)}:{inspect.getsourcelines(action_group_class)[1]}]{action_group_name}[/link]",
                f"[link=vscode://file/{action_group_path}]{action_group_relpath}[/link]",
            )
        print(table)

    # Print table for environments
    if RegisteredEntity.ENV in category:
        import gymnasium

        from srb.utils.registry import get_srb_tasks

        table = Table(title="Environments of the Space Robotics Bench")
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
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
            env_module_path = Path(
                inspect.getabsfile(
                    importlib.import_module(entrypoint_module.rsplit(".", 1)[0])
                )
            )
            try:
                env_module_relpath = env_module_path.parent.relative_to(
                    Path(inspect.getabsfile(srb_tasks)).parent
                )
            except ValueError:
                env_module_relpath = path.join("EXT", env_module_path.name)
            entrypoint_module = importlib.import_module(entrypoint_module)
            entrypoint_class = getattr(entrypoint_module, entrypoint_class)
            entrypoint_parent = entrypoint_class.__bases__[0]
            cfg_class = env.kwargs["task_cfg"]
            cfg_parent = cfg_class.__bases__[0]
            table.add_row(
                str(i),
                task_id.removeprefix("srb/")
                + (" (demo)" if "demo" in entrypoint_module.__name__ else ""),
                f"[link=vscode://file/{inspect.getabsfile(entrypoint_class)}:{inspect.getsourcelines(entrypoint_class)[1]}]{entrypoint_class.__name__}[/link]([red][link=vscode://file/{inspect.getabsfile(entrypoint_parent)}:{inspect.getsourcelines(entrypoint_parent)[1]}]{entrypoint_parent.__name__}[/link][/red])",
                f"[link=vscode://file/{inspect.getabsfile(cfg_class)}:{inspect.getsourcelines(cfg_class)[1]}]{cfg_class.__name__}[/link]([magenta][link=vscode://file/{inspect.getabsfile(cfg_parent)}:{inspect.getsourcelines(cfg_parent)[1]}]{cfg_parent.__name__}[/link][/magenta])",
                f"[link=vscode://file/{env_module_path}]{env_module_relpath}[/link]",
            )
        print(table)

    # Shutdown Isaac Sim
    launcher.app.close()


class RegisteredEntity(str, Enum):
    ALL = auto()
    ACTION = auto()
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
    )
    rand_agent_parser = agent_subparsers.add_parser(
        "rand",
        help="Random agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    teleop_agent_parser = agent_subparsers.add_parser(
        "teleop",
        help="Teleoperate agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ros_agent_parser = agent_subparsers.add_parser(
        "ros",
        help="ROS 2 or Space ROS agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_agent_parser = agent_subparsers.add_parser(
        "train",
        help="Train agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_agent_parser = agent_subparsers.add_parser(
        "eval",
        help="eval agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    collect_agent_parser = agent_subparsers.add_parser(
        "collect",
        help="Collect demonstrations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    learn_agent_parser = agent_subparsers.add_parser(
        "learn",
        help="Learn from demonstrations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    for _agent_parser in (
        zero_agent_parser,
        rand_agent_parser,
        teleop_agent_parser,
        ros_agent_parser,
        train_agent_parser,
        eval_agent_parser,
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

        video_recording_group = _agent_parser.add_argument_group("Video")
        video_recording_group.add_argument(
            "--video",
            action="store_true",
            default=False,
            help="Record videos",
        )
        video_recording_group.add_argument(
            "--video_length",
            type=int,
            default=1000,
            help="Length of the recorded video (in steps)",
        )
        video_recording_group.add_argument(
            "--video_interval",
            type=int,
            default=10000,
            help="Interval between video recordings (in steps)",
        )

        experience_group = _agent_parser.add_argument_group("Experience")
        experience_group.add_argument(
            "--hide_ui",
            action="store_true",
            default=False,
            help="Disable most of the Isaac Sim UI and set it to fullscreen",
        )

        launcher_group = _agent_parser.add_argument_group("Launcher")
        launcher_group.add_argument(
            "--headless",
            action="store_true",
            default=False,
            help="Run the simulation without display output",
        )
        launcher_group.add_argument(
            "--livestream",
            type=int,
            default=-1,
            choices={0, 1, 2},
            help="Force enable livestreaming. Mapping corresponds to that for the `LIVESTREAM` environment variable (0: Disabled, 1: Native, 2: WebRTC)",
        )
        launcher_group.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
            help="Compute device to use for simulation",
        )
        launcher_group.add_argument(
            "--kit_args",
            type=str,
            default="",
            help="CLI args for the Omniverse Kit as a string separated by a space delimiter (e.g., '--ext-folder=/path/to/ext1 --ext-folder=/path/to/ext2')",
        )

    for _agent_parser in (
        teleop_agent_parser,
        collect_agent_parser,
    ):
        teleop_group = _agent_parser.add_argument_group("Teleop")
        teleop_group.add_argument(
            "--teleop_device",
            type=str,
            nargs="+",
            choices=["keyboard", "gamepad", "joystick", "mouse", "ros2"],
            default=["keyboard"],  # TODO: Enum
            help="Device for interacting with environment",
        )
        teleop_group.add_argument(
            "--pos_sensitivity",
            type=float,
            default=10.0,
            help="Sensitivity factor for translation",
        )
        teleop_group.add_argument(
            "--rot_sensitivity",
            type=float,
            default=40.0,
            help="Sensitivity factor for rotation",
        )
        teleop_group.add_argument(
            "--disable_control_scheme_inversion",
            action="store_true",
            default=False,
            help="Flag to disable inverting the control scheme due to view for manipulation-based tasks",
        )

        integrations_group = _agent_parser.add_argument_group("Integrations")
        integrations_group.add_argument(
            "--integration",
            type=str,
            nargs="*",
            choices=["gui", "ros2"],
            default=[],  # TODO: Enum
            help="Sequence of integrations ro enable",
        )

    for _agent_parser in (
        train_agent_parser,
        eval_agent_parser,
        teleop_agent_parser,
        collect_agent_parser,
    ):
        algorithm_group = _agent_parser.add_argument_group(
            "Algorithm"
            if _agent_parser in (train_agent_parser, eval_agent_parser)
            else "Teleop Policy"
        )
        algorithm_group.add_argument(
            "--algo",
            type=str,
            help="Name of the algorithm",
            choices=[
                "dreamer",
                "skrl_ppo",
                "skrl_ppo_rnn",
                "sb3_ppo",
                "sb3_ppo_lstm",
                "sbx_ppo",
            ],  # TODO: Enum
            required=_agent_parser in (train_agent_parser, eval_agent_parser),
        )
        if _agent_parser != train_agent_parser:
            algorithm_group.add_argument(
                "--model",
                type=str,
                help="Path to the model",
            )

    train_group = train_agent_parser.add_argument_group("Train")
    mutex_group = train_group.add_mutually_exclusive_group()
    mutex_group.add_argument(
        "--continue_training",
        "--continue",
        "--resume",
        action="store_true",
        default=False,
        help="Continue training the model from the last checkpoint",
    )
    mutex_group.add_argument(
        "--model",
        type=str,
        help="Continue training the model from the specified checkpoint",
    )

    # GUI
    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    # Enable rich traceback (delayed after argcomplete to maintain snappy completion)
    from srb.utils.tracing import with_rich

    with_rich()

    # Allow separation of arguments meant for other purposes
    if "--" in sys.argv:
        sys.argv = [sys.argv[0], *sys.argv[(sys.argv.index("--") + 1) :]]

    # Parse arguments
    args, other_args = parser.parse_known_args()

    # Detect any unsupported arguments
    unsupported_args = [
        arg for arg in other_args if arg.startswith("-") or "=" not in arg
    ]
    if unsupported_args:
        import string

        raise ValueError(
            f'Unsupported CLI argument{"s" if len(unsupported_args) > 1 else ""}: '
            + ", ".join(
                f'"{arg}"' if any(c in string.whitespace for c in arg) else arg
                for arg in unsupported_args
            )
        )

    # Forward other arguments to hydra
    sys.argv = [sys.argv[0], *other_args]

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
