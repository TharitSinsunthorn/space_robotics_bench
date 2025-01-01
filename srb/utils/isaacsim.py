from importlib.util import find_spec


def is_isaacsim_initialized() -> bool:
    return find_spec("omni.isaac.version") is not None
