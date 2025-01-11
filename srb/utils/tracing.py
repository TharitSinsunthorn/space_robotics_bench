import importlib
from functools import cache
from importlib.util import find_spec
from os import environ


@cache
def with_rich() -> bool:
    if find_spec("rich") is None:
        return False

    from rich import traceback

    if environ.get("SRB_WITH_TRACEBACK", "true").lower() not in ("true", "1"):
        return False

    suppress = []
    for mod in (
        "numpy",
        "omegaconf",
        "pydantic",
        "torch",
    ):
        try:
            suppress.append(importlib.import_module(mod))
        except Exception:
            pass

    traceback.install(
        width=120,
        show_locals=environ.get("SRB_WITH_TRACEBACK_LOCALS", "false").lower()
        in ("true", "1"),
        suppress=suppress,
    )

    return True


@cache
def with_logfire() -> bool:
    if find_spec("logfire") is None:
        return False

    import logfire

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
