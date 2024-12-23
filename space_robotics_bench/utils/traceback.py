from os import environ


def enable_rich_traceback():
    try:
        from rich import traceback
    except ImportError:
        return

    if environ.get("SRB_WITH_TRACEBACK", "true").lower() not in ("true", "1"):
        return

    import numpy
    import torch

    traceback.install(
        width=120,
        show_locals=environ.get("SRB_WITH_TRACEBACK_LOCALS", "false").lower()
        in ("true", "1"),
        suppress=(numpy, torch),
    )

    # Disable traceback of SimForge
    environ["SF_WITH_TRACEBACK"] = "false"
