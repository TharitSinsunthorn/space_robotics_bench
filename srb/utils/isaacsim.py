from functools import cache
from importlib.util import find_spec
from os import environ
from pathlib import Path


@cache
def get_isaacsim_python() -> str:
    if isaac_sim_python := environ.get("ISAAC_SIM_PYTHON"):
        return isaac_sim_python

    standard_path = (
        Path(environ.get("HOME", "/root")).joinpath("isaac-sim").joinpath("python.sh")
    )
    if standard_path.exists():
        return standard_path.as_posix()

    return "python3"


def is_isaacsim_initialized() -> bool:
    return find_spec("isaacsim.core.version") is not None


def hide_isaacsim_ui():
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set("/app/window/hideUi", True)
    settings.set("/app/window/fullscreen", True)
