from os import environ

from srb.utils import logging
from srb.utils.importer import import_recursively
from srb.utils.isaacsim import is_isaacsim_initialized
from srb.utils.registry import get_srb_tasks
from srb.utils.tracing import with_logfire, with_rich

## Enable rich traceback
with_rich()

## Enable logfire instrumentation
with_logfire()

## If the simulation app is started, register all tasks by recursively importing the tasks submodule
if environ.get("SRB_SKIP_REGISTRATION", "false").lower() in ["true", "1"]:
    logging.info(
        f'Skipping the registration of the Space Robotics Bench tasks (SRB_SKIP_REGISTRATION={environ.get("SRB_SKIP_REGISTRATION")})'
    )
elif is_isaacsim_initialized():
    TASKS_MOD = f"{__name__}.tasks"
    import_recursively(module_name=TASKS_MOD)
    logging.debug(
        f"Recursively imported Space Robotics Bench module '{TASKS_MOD}' ({len(get_srb_tasks())} registered tasks)"
    )
else:
    logging.critical(
        "Tasks of the Space Robotics Bench cannot be registered because the simulation is not running. Please "
        "import the 'srb' module after starting the Omniverse simulation app. Alternatively, "
        "set the 'SRB_SKIP_REGISTRATION' environment variable to 'true' to skip the registration of tasks."
    )
