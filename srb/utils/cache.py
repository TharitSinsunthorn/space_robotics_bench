from typing import Sequence

from srb.utils.isaacsim import is_isaacsim_initialized
from srb.utils.path import SRB_ENV_CACHE_PATH


def read_env_list_cache() -> Sequence[str] | None:
    if not SRB_ENV_CACHE_PATH.exists():
        return None
    with SRB_ENV_CACHE_PATH.open("r") as f:
        return f.read().splitlines()


def update_env_list_cache():
    from srb.utils import logging
    from srb.utils.registry import SRB_NAMESPACE, get_srb_tasks

    if not is_isaacsim_initialized():
        logging.critical(
            "Updating the cache of registered environments will likely fail because Isaac Sim is not initialized"
        )

    from srb import tasks as _  # noqa: F401

    registered_envs = sorted(
        map(lambda env: env.removeprefix(f"{SRB_NAMESPACE}/"), get_srb_tasks())
    )

    if len(registered_envs) == 0:
        logging.warning(
            "Cannot update the cache of registered environments because no environments are registered"
        )
        return

    if registered_envs == read_env_list_cache():
        logging.trace("The cache of registered environments is up-to-date")
        return

    with SRB_ENV_CACHE_PATH.open("w") as f:
        f.write("\n".join(registered_envs) + "\n")
    logging.debug(
        f"Updated the cache of registered environments to {SRB_ENV_CACHE_PATH}"
    )
