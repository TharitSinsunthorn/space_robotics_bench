from os import environ

from .utils import (
    enable_rich_traceback,
    get_srb_tasks,
    import_modules_recursively,
    is_isaacsim_initialized,
    logging,
)

## Enable rich traceback
enable_rich_traceback()

## If the simulation app is started, register all tasks by recursively importing
## the "{__name__}.tasks" submodule
if environ.get("SRB_SKIP_REGISTRATION", "false").lower() in ["true", "1"]:
    logging.info(
        "Skipping the registration of the Space Robotics Bench tasks "
        f"(SRB_SKIP_REGISTRATION={environ.get('SRB_SKIP_REGISTRATION')})"
    )
elif is_isaacsim_initialized():
    # TODO: Revert
    import_modules_recursively(module_name=f"{__name__}.tasks._demos.cubesat")
    # import_modules_recursively(module_name=f"{__name__}.tasks")
    logging.info(
        f"Registered '{len(get_srb_tasks())}' Gymnasium tasks of the Space Robotics Bench"
    )
else:
    raise RuntimeError(
        "Tasks of the Space Robotics Bench cannot be registered because the simulation "
        "is not running. Please import the 'space_robotics_bench' module after starting the "
        "Omniverse simulation app. Alternatively, set the 'SRB_SKIP_REGISTRATION' environment "
        "variable to 'true' to skip the registration of tasks."
    )
