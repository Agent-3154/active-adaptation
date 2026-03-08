"""Isaac Sim / Omniverse keyboard input manager for teleop."""

import weakref
from collections import defaultdict


class IsaacKeyboardManager:
    """Manages keyboard state via Omniverse carb input. Use for Isaac backend teleop."""

    def __init__(self) -> None:
        import carb
        import omni.appwindow

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self.key_pressed = defaultdict(lambda: False)

        def on_keyboard_event(event, *args):
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self.key_pressed[event.input.name] = True
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self.key_pressed[event.input.name] = False

        self.on_keyboard_event = on_keyboard_event
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj.on_keyboard_event(event, *args),
        )
