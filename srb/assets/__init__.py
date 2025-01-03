from srb.utils import logging
from srb.utils.importer import import_recursively
from srb.utils.isaacsim import is_isaacsim_initialized
from srb.utils.registry import get_srb_tasks

from .light import *  # noqa: F403
from .object import *  # noqa: F403
from .robot import *  # noqa: F403
from .terrain import *  # noqa: F403

# TODO: Decide if this is needed


if is_isaacsim_initialized():
    import_recursively(__name__)
    logging.debug(
        f'Recursively imported Space Robotics Bench module "{__name__}" ({len(get_srb_tasks())} registered assets)'
    )
else:
    raise RuntimeError(
        "Assets of the Space Robotics Bench cannot be registered because Isaac Sim is not initialized. "
        f'Please import the "{__name__}" module after starting the Omniverse simulation app.'
    )
