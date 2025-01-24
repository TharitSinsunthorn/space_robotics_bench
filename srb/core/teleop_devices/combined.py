import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Sequence

import numpy

from srb.core.actions import (
    ManipulatorTaskSpaceActionCfg,
    MultiCopterActionGroupCfg,
    SpacecraftActionGroupCfg,
    WheeledRoverActionGroupCfg,
)
from srb.core.teleop_devices import DeviceBase, Se3Gamepad
from srb.core.teleop_devices.keyboard import KeyboardTeleopInterface
from srb.core.teleop_devices.spacemouse import SpacemouseTeleopInterface

if TYPE_CHECKING:
    from rclpy.node import Node

# TODO: Use enum for devices


class CombinedTeleopInterface(DeviceBase):
    def __init__(
        self,
        devices: Sequence[str],
        node: "Node | None" = None,
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        action_cfg: ManipulatorTaskSpaceActionCfg
        | MultiCopterActionGroupCfg
        | WheeledRoverActionGroupCfg
        | None = None,
    ):
        if not node and ("ros2" in devices or "haptic" in devices):
            from srb.utils.ros2 import enable_ros2_bridge

            enable_ros2_bridge()
            import rclpy
            from rclpy.node import Node

            rclpy.init(args=None)
            self._node = Node("srb")  # type: ignore
        else:
            self._node = node

        self._action_cfg = action_cfg
        self.interfaces = []
        self.ft_feedback_interfaces = []
        for device in devices:
            match device.lower():
                case "keyboard":
                    self.interfaces.append(
                        KeyboardTeleopInterface(
                            pos_sensitivity=0.05 * pos_sensitivity,
                            rot_sensitivity=10.0 * rot_sensitivity,
                        )
                    )
                case "spacemouse":
                    self.interfaces.append(
                        SpacemouseTeleopInterface(
                            pos_sensitivity=0.1 * pos_sensitivity,
                            rot_sensitivity=0.05 * rot_sensitivity,
                        )
                    )
                case "gamepad":
                    self.interfaces.append(
                        Se3Gamepad(
                            pos_sensitivity=0.1 * pos_sensitivity,
                            rot_sensitivity=0.1 * rot_sensitivity,
                        )
                    )
                case "ros2":
                    from srb.core.teleop_devices.ros2 import ROS2TeleopInterface

                    self.interfaces.append(
                        ROS2TeleopInterface(
                            node=self._node,
                            pos_sensitivity=1.0 * pos_sensitivity,
                            rot_sensitivity=1.0 * rot_sensitivity,
                        )
                    )
                case "haptic":
                    from srb.core.teleop_devices.haptic import HapticROS2TeleopInterface

                    interface = HapticROS2TeleopInterface(
                        node=self._node,
                        pos_sensitivity=1.0 * pos_sensitivity,
                        rot_sensitivity=0.15 * rot_sensitivity,
                    )
                    self.interfaces.append(interface)
                    self.ft_feedback_interfaces.append(interface)
                case _:
                    raise ValueError(f"Invalid device interface '{device}'.")

            self.gain = 1.0

            def cb_gain_decrease():
                self.gain *= 0.75
                print(f"Gain: {self.gain}")

            self.add_callback("O", cb_gain_decrease)

            def cb_gain_increase():
                self.gain *= 1.25
                print(f"Gain: {self.gain}")

            self.add_callback("P", cb_gain_increase)

        # Run a thread for listening to device
        if not node and self._node is not None:
            self._thread = threading.Thread(target=rclpy.spin, args=(self._node,))
            self._thread.daemon = True
            self._thread.start()

    def __del__(self):
        for interface in self.interfaces:
            interface.__del__()

    def __str__(self) -> str:
        from srb.core.teleop_devices.keyboard import KeyboardTeleopInterface

        msg = "Combined Interface\n"
        msg += f"Devices: {', '.join([interface.__class__.__name__ for interface in self.interfaces])}\n"

        for interface in self.interfaces:
            if (
                isinstance(interface, KeyboardTeleopInterface)
                and self._action_cfg is not None
            ):
                msg += self._keyboard_control_scheme(self._action_cfg)
                continue
            msg += "\n"
            msg += interface.__str__()

        return msg

    """
    Operations
    """

    def reset(self):
        for interface in self.interfaces:
            interface.reset()

        self._close_gripper = False
        self._prev_gripper_cmds = [False] * len(self.interfaces)

    def add_callback(self, key: str, func: Callable):
        for interface in self.interfaces:
            if isinstance(interface, KeyboardTeleopInterface):
                interface.add_callback(key=key, func=func)
            if isinstance(interface, SpacemouseTeleopInterface) and key in [
                "L",
                "R",
                "LR",
            ]:
                interface.add_callback(key=key, func=func)

    def advance(self) -> tuple[numpy.ndarray, bool]:
        raw_actions = [interface.advance() for interface in self.interfaces]

        twist = self.gain * numpy.sum(
            numpy.stack([a[0] for a in raw_actions], axis=0), axis=0
        )

        for i, prev_gripper_cmd in enumerate(self._prev_gripper_cmds):
            if prev_gripper_cmd != raw_actions[i][1]:
                self._close_gripper = not self._close_gripper
                break
        self._prev_gripper_cmds = [a[1] for a in raw_actions]

        return twist, self._close_gripper

    def set_ft_feedback(self, ft_feedback: numpy.ndarray):
        for interface in self.ft_feedback_interfaces:
            interface.set_ft_feedback(ft_feedback)

    @staticmethod
    def _keyboard_control_scheme(
        action_cfg: ManipulatorTaskSpaceActionCfg
        | MultiCopterActionGroupCfg
        | WheeledRoverActionGroupCfg,
    ) -> str:
        if isinstance(action_cfg, ManipulatorTaskSpaceActionCfg):
            return """
+------------------------------------------------+
|  Keyboard Scheme (focus the Isaac Sim window)  |
+------------------------------------------------+
+------------------------------------------------+
| Reset: [ L ]                                   |
| Decrease Gain [ O ]   | Increase Gain: [ P ]   |
| Toggle Gripper: [ R / K ]                      |
+------------------------------------------------+
| Translation                                    |
|             [ W ] (+X)            [ Q ] (+Z)   |
|               ↑                     ↑          |
|               |                     |          |
|  (-Y) [ A ] ← + → [ D ] (+Y)        +          |
|               |                     |          |
|               ↓                     ↓          |
|             [ S ] (-X)            [ E ] (-Z)   |
|------------------------------------------------|
| Rotation                                       |
|       [ Z ] ←--------(±X)--------→ [ X ]       |
|                                                |
|       [ T ] ↻--------(±Y)--------↺ [ G ]       |
|                                                |
|       [ C ] ↺--------(±Z)--------↻ [ V ]       |
+------------------------------------------------+
        """
        elif isinstance(action_cfg, MultiCopterActionGroupCfg):
            return """
+------------------------------------------------+
|  Keyboard Scheme (focus the Isaac Sim window)  |
+------------------------------------------------+
+------------------------------------------------+
| Decrease Gain [ O ]   | Increase Gain: [ P ]   |
| Reset: [ L ]                                   |
+------------------------------------------------+
|                  Translation                   |
|             [ W ] (+X)            [ Q ] (+Z)   |
|               ↑                     ↑          |
|               |                     |          |
|  (-Y) [ A ] ← + → [ D ] (+Y)        +          |
|               |                     |          |
|               ↓                     ↓          |
|             [ S ] (-X)            [ E ] (-Z)   |
|------------------------------------------------|
|                    Rotation                    |
|       [ C ] ↺--------(±Z)--------↻ [ V ]       |
+------------------------------------------------+
        """
        elif isinstance(action_cfg, WheeledRoverActionGroupCfg):
            return """
+------------------------------------------------+
|  Keyboard Scheme (focus the Isaac Sim window)  |
+------------------------------------------------+
+------------------------------------------------+
| Decrease Gain [ O ]   | Increase Gain: [ P ]   |
| Reset: [ L ]                                   |
+------------------------------------------------+
| Planar Motion                                  |
|                     [ W ] (+X)                 |
|                       ↑                        |
|                       |                        |
|          (-Y) [ A ] ← + → [ D ] (+Y)           |
|                       |                        |
|                       ↓                        |
|                     [ S ] (-X)                 |
+------------------------------------------------+
        """
        elif isinstance(action_cfg, SpacecraftActionGroupCfg):
            return """
+------------------------------------------------+
|  Keyboard Scheme (focus the Isaac Sim window)  |
+------------------------------------------------+
+------------------------------------------------+
| Decrease Gain [ O ]   | Increase Gain: [ P ]   |
| Reset: [ L ]                                   |
+------------------------------------------------+
| Translation                                    |
|             [ W ] (+X)            [ Q ] (+Z)   |
|               ↑                     ↑          |
|               |                     |          |
|  (-Y) [ A ] ← + → [ D ] (+Y)        +          |
|               |                     |          |
|               ↓                     ↓          |
|             [ S ] (-X)            [ E ] (-Z)   |
|------------------------------------------------|
| Rotation                                       |
|       [ Z ] ←--------(±X)--------→ [ X ]       |
|                                                |
|       [ T ] ↻--------(±Y)--------↺ [ G ]       |
|                                                |
|       [ C ] ↺--------(±Z)--------↻ [ V ]       |
+------------------------------------------------+
        """
        else:
            return """
+------------------------------------------------+
|  Keyboard Scheme (focus the Isaac Sim window)  |
+------------------------------------------------+
+------------------------------------------------+
| Reset: [ L ]                                   |
| Decrease Gain [ O ]   | Increase Gain: [ P ]   |
| Event: [ R / K ]                               |
+------------------------------------------------+
| Translation                                    |
|             [ W ] (+X)            [ Q ] (+Z)   |
|               ↑                     ↑          |
|               |                     |          |
|  (-Y) [ A ] ← + → [ D ] (+Y)        +          |
|               |                     |          |
|               ↓                     ↓          |
|             [ S ] (-X)            [ E ] (-Z)   |
|------------------------------------------------|
| Rotation                                       |
|       [ Z ] ←--------(±X)--------→ [ X ]       |
|                                                |
|       [ T ] ↻--------(±Y)--------↺ [ G ]       |
|                                                |
|       [ C ] ↺--------(±Z)--------↻ [ V ]       |
+------------------------------------------------+
        """
