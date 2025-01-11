import enum
import functools
from collections.abc import Callable

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    raise ImportError(
        "Hydra is not installed. Please install it by running 'pip install hydra-core'."
    )

import collections.abc
from typing import Any, Iterable, Mapping, get_type_hints

from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from omni.isaac.lab.envs.utils.spaces import (
    replace_env_cfg_spaces_with_strings,
    replace_strings_with_env_cfg_spaces,
)
from pydantic import BaseModel

from srb.utils import logging
from srb.utils.dict import replace_slices_with_strings, replace_strings_with_slices
from srb.utils.parsing import load_cfg_from_registry

# TODO: Combine with cfg and parsing modules


def register_task_to_hydra(
    task_name: str, agent_cfg_entry_point: str | None = None
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, dict]:
    """Register the task configuration to the Hydra configuration store.

    This function resolves the configuration file for the environment and agent based on the task's name.
    It then registers the configurations to the Hydra configuration store.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        A tuple containing the parsed environment and agent configuration objects.
    """
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
    """Decorator to handle the Hydra configuration for a task.

    This decorator registers the task to Hydra and updates the environment and agent configurations from Hydra parsed
    command line arguments.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        The decorated function with the envrionment's and agent's configurations updated from command line arguments.
    """

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


# TODO: Clean-up


def reconstruct_object(obj: Any, updates: Mapping[str, Any]) -> Any:
    """
    Reconstruct an object, applying updates. Handles various types, including functions.

    Args:
        obj: The object to reconstruct.
        updates: Dictionary of updates.
        visited: Set of visited object IDs to detect circular references.

    Returns:
        A new object instance with applied updates, or the original callable for functions.
    """

    try:
        if isinstance(obj, BaseModel):
            type_hints = get_type_hints(obj.__class__)
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
            type_hints = get_type_hints(obj.__class__)
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
