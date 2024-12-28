from functools import cache
from os import environ


@cache
def with_rich() -> bool:
    try:
        from rich import traceback
    except ImportError:
        return False

    if environ.get("SRB_WITH_TRACEBACK", "true").lower() not in ("true", "1"):
        return False

    import numpy
    import pydantic
    import torch

    traceback.install(
        width=120,
        show_locals=environ.get("SRB_WITH_TRACEBACK_LOCALS", "false").lower()
        in ("true", "1"),
        suppress=(numpy, pydantic, torch),
    )

    return True


@cache
def with_logfire() -> bool:
    try:
        import logfire
    except ImportError:
        return False

    if environ.get("SRB_WITH_LOGFIRE", "true").lower() not in ("true", "1"):
        return False

    logfire.configure(
        send_to_logfire=environ.get("LOGFIRE_SEND_TO_LOGFIRE", "false").lower()
        in ("true", "1")
        or environ.get("SRB_WITH_LOGFIRE_SEND", "false").lower() in ("true", "1"),
        service_name="srb",
        console=False,
    )
    logfire.instrument_pydantic()

    return True
