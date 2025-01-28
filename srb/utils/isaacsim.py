from importlib.util import find_spec


def is_isaacsim_initialized() -> bool:
    return find_spec("omni.isaac.version") is not None


def hide_isaacsim_ui():
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set("/app/window/hideUi", True)
    settings.set("/app/window/fullscreen", True)
