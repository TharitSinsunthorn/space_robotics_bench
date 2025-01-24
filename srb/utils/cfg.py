import collections.abc
import datetime
import enum
import functools
import importlib
import inspect
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple, get_type_hints

import gymnasium
import hydra
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from srb._typing import AnyEnvCfg
from srb.utils import logging
from srb.utils.dict import (
    replace_slices_with_strings,
    replace_strings_with_slices,
    update_class_from_dict,
    update_dict,
)
from srb.utils.spaces import (
    replace_env_cfg_spaces_with_strings,
    replace_strings_with_env_cfg_spaces,
)

SUPPORTED_FRAMEWORKS = {
    "dreamer": {"multi_algo": False},
    "sb3": {"multi_algo": True},
    "sbx": {"multi_algo": True},
    "skrl": {"multi_algo": True},
    "robomimic": {"multi_algo": True},
}
SUPPORTED_CFG_FILE_EXTENSIONS = (
    "json",
    "toml",
    "yaml",
    "yml",
)
FRAMEWORK_CFG_ENTRYPOINT_KEY = "{FRAMEWORK}_cfg"
FRAMEWORK_MULTI_ALGO_CFG_ENTRYPOINT_KEY = "{FRAMEWORK}_{ALGO}_cfg"


def parse_algo_configs(cfg_dir: str) -> Mapping[str, str]:
    algo_config = {}

    for root, _, files in os.walk(cfg_dir):
        for file in files:
            if not file.endswith(SUPPORTED_CFG_FILE_EXTENSIONS):
                continue
            file = os.path.join(root, file)

            key = _identify_config(root, file)
            if key is not None:
                algo_config[key] = file

    return algo_config


def _identify_config(root: str, file) -> str | None:
    basename = os.path.basename(file).split(".")[0]

    for framework, properties in SUPPORTED_FRAMEWORKS.items():
        algo = basename.replace(f"{framework}_", "")
        if root.endswith(framework):
            assert properties["multi_algo"]
            return FRAMEWORK_MULTI_ALGO_CFG_ENTRYPOINT_KEY.format(
                FRAMEWORK=framework, ALGO=algo
            )
        elif basename.startswith(f"{framework}"):
            if properties["multi_algo"]:
                return FRAMEWORK_MULTI_ALGO_CFG_ENTRYPOINT_KEY.format(
                    FRAMEWORK=framework, ALGO=algo
                )
            else:
                return FRAMEWORK_CFG_ENTRYPOINT_KEY.format(FRAMEWORK=framework)

    return None


def load_cfg_from_registry(
    task_name: str, entry_point_key: str, unpack_callable: bool = True
) -> AnyEnvCfg | Dict[str, Any]:
    # Obtain the configuration entry point
    cfg_entry_point = gymnasium.spec(task_name).kwargs.get(entry_point_key)
    # Check if entry point exists
    if cfg_entry_point is None:
        raise ValueError(
            f"Could not find configuration for the environment: '{task_name}'."
            f" Please check that the gym registry has the entry point: '{entry_point_key}'."
            f" Found: {gymnasium.spec(task_name).kwargs}."
        )
    # Parse the default config file
    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if os.path.exists(cfg_entry_point):
            # Absolute path for the config file
            config_file = cfg_entry_point
        else:
            # Resolve path to the module location
            mod_name, file_name = cfg_entry_point.split(":")
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
            # Obtain the configuration file path
            config_file = os.path.join(mod_path, file_name)
        # Load the configuration
        print(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.full_load(f)
    else:
        if unpack_callable and callable(cfg_entry_point):
            # Resolve path to the module location
            mod_path = inspect.getfile(cfg_entry_point)
            # Load the configuration
            cfg_cls = cfg_entry_point()
        elif isinstance(cfg_entry_point, str):
            # Resolve path to the module location
            mod_name, attr_name = cfg_entry_point.split(":")
            mod = importlib.import_module(mod_name)
            cfg_cls = getattr(mod, attr_name)
        else:
            cfg_cls = cfg_entry_point
        # Load the configuration
        print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")
        cfg = cfg_cls() if unpack_callable and callable(cfg_cls) else cfg_cls
    return cfg


def parse_task_cfg(
    task_name: str,
    device: str = "cuda:0",
    num_envs: int | None = None,
    use_fabric: bool | None = None,
) -> AnyEnvCfg | Dict[str, Any]:
    # Create a dictionary to update from
    args_cfg = {"sim": {}, "scene": {}}

    # Simulation device
    args_cfg["sim"]["device"] = device

    # Disable fabric to read/write through USD
    if use_fabric is not None:
        args_cfg["sim"]["use_fabric"] = use_fabric

    # Number of environments
    if num_envs is not None:
        args_cfg["scene"]["num_envs"] = num_envs

    # Load the default configuration
    cfg = load_cfg_from_registry(task_name, "task_cfg", unpack_callable=False)
    # Update the main configuration
    if callable(cfg):
        default_cfg = cfg()
        cfg = cfg(
            sim=default_cfg.sim.replace(**args_cfg["sim"]),
            scene=default_cfg.scene.replace(**args_cfg["scene"]),
        )
    elif isinstance(cfg, dict):
        cfg = update_dict(cfg, args_cfg)
    else:
        update_class_from_dict(cfg, args_cfg)

    return cfg


def create_logdir_path(
    algo_name: str,
    task_name: str,
    prefix: str = "logs/",
    timestamp_format="%Y%m%d-%H%M%S",
) -> str:
    timestamp = datetime.datetime.now().strftime(timestamp_format)
    logdir = os.path.realpath(os.path.join(prefix, algo_name, task_name, timestamp))
    os.makedirs(logdir, exist_ok=True)
    return logdir


def get_last_run_logdir_path(
    algo_name: str,
    task_name: str,
    prefix: str = "logs/",
) -> str:
    logdir_root = os.path.abspath(os.path.join(prefix, algo_name, task_name))
    logdirs = [
        os.path.join(logdir_root, d)
        for d in os.listdir(logdir_root)
        if os.path.isdir(os.path.join(logdir_root, d))
    ]
    logdirs.sort(key=os.path.getmtime, reverse=True)
    if len(logdirs) == 0:
        raise FileNotFoundError(f"No logdirs found in: {logdir_root}")
    last_logdir = None
    for d in logdirs:
        if not d.endswith("eval"):
            last_logdir = d
            break
    if last_logdir is None:
        raise FileNotFoundError(f"No logdirs found in: {logdir_root}")
    return last_logdir


def get_last_file(
    path: str | Path,
) -> Path:
    logdirs = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if os.path.isfile(os.path.join(path, d))
    ]
    logdirs.sort(key=os.path.getmtime, reverse=True)
    if len(logdirs) == 0:
        raise FileNotFoundError(f"No logdirs found in: {path}")
    last_logdir = None
    for d in logdirs:
        if not d.endswith("eval"):
            last_logdir = d
            break
    return Path(last_logdir)


def get_last_dir(
    path: str | Path,
) -> Path:
    logdirs = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]
    logdirs.sort(key=os.path.getmtime, reverse=True)
    if len(logdirs) == 0:
        raise FileNotFoundError(f"No logdirs found in: {path}")
    last_logdir = None
    for d in logdirs:
        if not d.endswith("eval"):
            last_logdir = d
            break
    return Path(last_logdir)


def register_task_to_hydra(
    task_name: str, agent_cfg_entry_point: str | None = None
) -> Tuple[AnyEnvCfg, Dict[str, Any]]:
    # load the configurations
    env_cfg = load_cfg_from_registry(task_name, "task_cfg")
    # replace gymnasium spaces with strings because OmegaConf does not support them.
    # this must be done before converting the env configs to dictionary to avoid internal reinterpretations
    replace_env_cfg_spaces_with_strings(env_cfg)
    # convert the configs to dictionary
    env_cfg_dict = env_cfg.to_dict()

    if agent_cfg_entry_point is None:
        agent_cfg = {}
        agent_cfg_dict = {}
    else:
        agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point)
        if isinstance(agent_cfg, dict):
            agent_cfg_dict = agent_cfg
        else:
            agent_cfg_dict = agent_cfg.to_dict()
    cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict}
    # replace slices with strings because OmegaConf does not support slices
    cfg_dict = replace_slices_with_strings(cfg_dict)
    # store the configuration to Hydra
    ConfigStore.instance().store(name=task_name.rsplit("/", 1)[1], node=cfg_dict)
    return env_cfg, agent_cfg


def hydra_task_config(
    task_name: str, agent_cfg_entry_point: str | None = None
) -> Callable:
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # register the task to Hydra
            env_cfg, agent_cfg = register_task_to_hydra(
                task_name, agent_cfg_entry_point
            )

            # define the new Hydra main function
            @hydra.main(
                config_path=None,
                config_name=task_name.rsplit("/", 1)[1],
                version_base="1.3",
            )
            def hydra_main(
                hydra_env_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg
            ):
                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                # update the configs with the Hydra command line arguments
                # env_cfg.from_dict(hydra_env_cfg["env"])
                env_cfg = reconstruct_object(env_cfg, hydra_env_cfg["env"])
                # replace strings that represent gymnasium spaces because OmegaConf does not support them.
                # this must be done after converting the env configs from dictionary to avoid internal reinterpretations
                replace_strings_with_env_cfg_spaces(env_cfg)
                # get agent configs
                if isinstance(agent_cfg, dict):
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    # agent_cfg.from_dict(hydra_env_cfg["agent"])
                    agent_cfg = reconstruct_object(agent_cfg, hydra_env_cfg["agent"])
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

            # call the new Hydra main function
            hydra_main()

        return wrapper

    return decorator


def reconstruct_object(obj: Any, updates: Mapping[str, Any]) -> Any:
    try:
        if isinstance(obj, BaseModel):
            try:
                type_hints = get_type_hints(obj.__class__)
            except Exception:
                type_hints = {k: type(v) for k, v in obj.__dict__.items()}
            new_kwargs = {}
            for field_name, field_type in type_hints.items():
                if field_name.startswith("_"):
                    continue
                current_value = getattr(obj, field_name, None)
                update_value = updates.get(field_name, None)
                if update_value is not None:
                    new_kwargs[field_name] = reconstruct_object(
                        current_value, update_value
                    )
                else:
                    new_kwargs[field_name] = current_value

            return obj.__class__(**new_kwargs)

        if isinstance(obj, enum.Enum):
            if isinstance(updates, str):
                return obj.__class__[updates.strip().upper()]
            if isinstance(updates, Mapping) and "_name_" in updates.keys():
                return obj.__class__[updates["_name_"]]
            # Handle enums with NONE value
            if updates is None and hasattr(obj, "NONE"):
                return obj.__class__.NONE

        # Handle primitive and immutable types (strings, integers, etc.)
        if isinstance(obj, (str, int, float, bool, type(None))):
            return updates if updates is not None else obj

        # Handle callable objects (e.g., functions)
        if callable(obj):
            # Return the original function if it doesn't require reconstruction
            return obj

        # Handle mappings (e.g., dictionaries)
        if isinstance(obj, Mapping):
            result = obj.__class__(
                (
                    key,
                    reconstruct_object(obj.get(key, None), updates.get(key, None)),
                )
                for key in set(obj) | set(updates)
            )
            return result

        # Handle iterables (e.g., lists, tuples)
        if isinstance(obj, collections.abc.Iterable) and not isinstance(
            obj, (str, bytes)
        ):
            if not isinstance(updates, Iterable):
                raise ValueError(
                    f"Incompatible update type for iterable: {type(updates)}"
                )
            result = obj.__class__(
                reconstruct_object(o, u) for o, u in zip(obj, updates)
            )
            return result

        # Handle dataclasses and objects with attributes
        if hasattr(obj, "__dict__") or hasattr(obj, "__dataclass_fields__"):
            try:
                type_hints = get_type_hints(obj.__class__)
            except Exception:
                type_hints = {k: type(v) for k, v in obj.__dict__.items()}
            new_kwargs = {}
            for field_name, field_type in type_hints.items():
                if field_name.startswith("_"):
                    continue
                current_value = getattr(obj, field_name, None)
                update_value = updates.get(field_name, None)
                if update_value is not None:
                    new_kwargs[field_name] = reconstruct_object(
                        current_value, update_value
                    )
                else:
                    new_kwargs[field_name] = current_value

            return obj.__class__(**new_kwargs)

        # In case the object doesn't match any of the known types, return it directly
        return updates if updates is not None else obj
    except Exception as e:
        logging.error(
            f"Reconstruction error\n"
            f"\tobject type: {type(obj)}\n"
            f"\tobject value: {obj}\n"
            f"\tupdates type: {type(updates)}\n"
            f"\tupdates value: {updates}\n"
            f"\texception: {e}\n"
        )
        return obj
