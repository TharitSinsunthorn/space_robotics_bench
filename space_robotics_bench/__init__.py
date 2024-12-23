from os import environ

from .utils.importer import import_modules_recursively
from .utils.sim_app import is_sim_app_started
from .utils.traceback import try_enable_rich_traceback

## Try enabling rich traceback
try_enable_rich_traceback()

## Verify the availability of the Rust extension module
try:
    from . import _rs  # noqa: F401
except Exception:
    raise ModuleNotFoundError(from space_robotics_bench._rs.utils.sampling import *  # noqa: F403
        "Failed to import Python submodule 'space_robotics_bench._rs' that contains the Rust "
        "extension module. Please ensure that the package has been installed correctly."
    )

## If the simulation app is started, register all tasks by recursively importing
## the "{__name__}.tasks" submodule
if environ.get("SRB_SKIP_REGISTRATION", "false").lower() in ["true", "1"]:
    print(
        f"INFO: [SRB_SKIP_REGISTRATION={environ.get('SRB_SKIP_REGISTRATION')}] Skipping "
        "the registration of the Space Robotics Bench tasks."
    )
elif is_sim_app_started():
    import_modules_recursively(
        module_name=f"{__name__}.tasks",
        ignorelist=[
            "common",
        ],
    )
else:
    raise RuntimeError(
        "Tasks of the Space Robotics Bench cannot be registered because the simulation "
        "is not running. Please import the 'space_robotics_bench' module after starting the "
        "Omniverse simulation app. Alternatively, set the 'SRB_SKIP_REGISTRATION' environment "
        "variable to 'true' to skip the registration of tasks."
    )
