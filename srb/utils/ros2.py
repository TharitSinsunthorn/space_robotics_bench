from functools import cache
from importlib.util import find_spec
from os import environ
from pathlib import Path

import omni.ext
import omni.kit.app

from srb.utils import logging


@cache
def enable_ros2_bridge():
    if find_spec("rclpy") is not None:
        logging.debug(
            'ROS 2 is already sourced in the current environment, so "ros2_bridge" does not need to be enabled'
        )
        return

    # Include the internal ROS 2 libraries in the LD_LIBRARY_PATH
    INTERNAL_ROS_DISTRO: str = "humble"
    ld_library_path = environ.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        ld_library_path = f":{ld_library_path}".replace("::", ":")
    ros2_lib_path = Path(
        f'{environ.get("ISAAC_PATH")}/exts/omni.isaac.ros2_bridge/{INTERNAL_ROS_DISTRO}/lib'
    )
    assert ros2_lib_path.exists()
    environ["LD_LIBRARY_PATH"] = ros2_lib_path.as_posix() + ld_library_path

    # Get the extension manager and list of available extensions
    extension_manager = omni.kit.app.get_app().get_extension_manager()

    # Load the ROS extension
    for ext in extension_manager.get_extensions():
        if "omni.isaac.ros2_bridge-" not in ext["id"]:
            continue
        logging.debug('Enabling extension "{}"'.format(ext["id"]))
        extension_manager.set_extension_enabled_immediate(ext["id"], True)

    if find_spec("rclpy") is None:
        logging.error(
            'ROS 2 Python client library "rclpy" is still not available after trying to enable the "ros2_bridge" extension'
        )
