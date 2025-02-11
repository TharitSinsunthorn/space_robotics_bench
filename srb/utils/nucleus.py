from pathlib import Path

from isaaclab.utils.assets import (  # noqa:F401
    ISAAC_NUCLEUS_DIR,
    ISAACLAB_NUCLEUS_DIR,
    NUCLEUS_ASSET_ROOT_DIR,
    NVIDIA_NUCLEUS_DIR,
)

from srb.utils import logging


def get_local_or_nucleus_path(local_path: Path, nucleus_path: str) -> str:
    if local_path.exists():
        return local_path.as_posix()
    logging.debug(
        f"Falling back to nucleus path {nucleus_path} because {local_path} does not exist"
    )
    return nucleus_path
