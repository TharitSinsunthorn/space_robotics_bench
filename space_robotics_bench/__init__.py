from os import environ

from .utils import get_srb_tasks, import_recursively, is_isaacsim_initialized, logging
from .utils.tracing import with_logfire, with_rich

## Enable rich traceback
with_rich()

## Enable logfire instrumentation
with_logfire()


## If the simulation app is started, register all tasks by recursively importing
## the "{__name__}.tasks" submodule
if environ.get("SRB_SKIP_REGISTRATION", "false").lower() in ["true", "1"]:
    logging.info(
        "Skipping the registration of the Space Robotics Bench tasks "
        f"(SRB_SKIP_REGISTRATION={environ.get('SRB_SKIP_REGISTRATION')})"
    )
elif is_isaacsim_initialized():
    import_recursively(module_name=f"{__name__}.tasks")
    logging.debug(
        f"Registered Gymnasium tasks of the Space Robotics Bench ({len(get_srb_tasks())} registered tasks)"
    )
else:
    raise RuntimeError(
        "Tasks of the Space Robotics Bench cannot be registered because the simulation "
        "is not running. Please import the 'space_robotics_bench' module after starting the "
        "Omniverse simulation app. Alternatively, set the 'SRB_SKIP_REGISTRATION' environment "
        "variable to 'true' to skip the registration of tasks."
    )
