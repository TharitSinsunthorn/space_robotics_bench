import carb
from omni.isaac.lab.devices import Se3Keyboard as __Se3Keyboard


class Se3Keyboard(__Se3Keyboard):
    def __str__(self) -> str:
        msg = super().__str__()

        msg += "\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tAdditional controls:\n"
        msg += "\tToggle gripper (alternative): R\n"
        return msg

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        ret = super()._on_keyboard_event(event, *args, **kwargs)

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:  # type: ignore
            if event.input.name == "R":
                self._close_gripper = not self._close_gripper

        return ret
