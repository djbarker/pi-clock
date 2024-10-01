import numpy as np
import logging

from abc import ABC, abstractmethod
from colorsys import hsv_to_rgb
from PIL import ImageDraw, Image
from typing import Callable, Literal

# Raspberry Pi libs
from gpiozero import Button
from displayhatmini import DisplayHATMini as _DisplayHATMini
from unicornhatmini import UnicornHATMini as _UnicornHATMini


from utils import cmyk_to_rgb


__all__ = ["Screen", "UnicornHatMini"]

log = logging.getLogger(__name__)

ButtonT = Literal["a", "b", "x", "y"]


class Screen(ABC):

    @abstractmethod
    def set_brightness(self, brightness: float):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def set_pixel_rgb(self, x: int, y: int, r: int, g: int, b: int):
        pass

    @abstractmethod
    def set_button_handler(self, button: ButtonT, func: Callable[[], None]):
        pass

    def set_pixel_cmyk(self, x: int, y: int, c: int, y_: int, m: int, k: int):
        r, g, b = cmyk_to_rgb(c, m, y_, k)
        self.set_pixel_rgb(x, y, r, g, b)

    def set_all_rgb(self, img: np.ndarray):
        # assert img.dims == 3
        assert img.shape[0] == self.width
        assert img.shape[1] == self.height
        assert img.shape[2] == 3

        for x in range(self.width):
            for y in range(self.height):
                r = img[x, y, 0]
                g = img[x, y, 1]
                b = img[x, y, 2]
                self.set_pixel_rgb(x, y, r, g, b)

    def set_all_hsv(self, img: np.ndarray):
        # assert img.dims == 3
        assert img.shape[0] == self.width
        assert img.shape[1] == self.height
        assert img.shape[2] == 3

        for x in range(self.width):
            for y in range(self.height):
                h = img[x, y, 0]
                s = img[x, y, 1]
                v = img[x, y, 2]
                r, g, b = hsv_to_rgb(h, s, v)
                r = int(255 * r)
                g = int(255 * g)
                b = int(255 * b)
                self.set_pixel_rgb(x, y, r, g, b)


class UnicornHatMini(Screen):
    """
    See: https://github.com/pimoroni/unicornhatmini-python
    """

    def __init__(self, brightness: float = 0.25, flip: bool = False):
        self.uh = _UnicornHATMini()

        # TODO: what does this do? Can I remove flip
        rotation = 0
        self.uh.set_rotation(rotation)

        self.width, self.height = self.uh.get_shape()

        self.uh.set_brightness(brightness)
        self.flip = flip

        self.buttons = {
            "a": Button(5),
            "b": Button(6),
            "x": Button(16),
            "y": Button(24),
        }

    def set_brightness(self, brightness: float):
        log.debug(f"Screen brightness set to {brightness:.2f}")
        brightness = min(max(brightness, 0.02), 1.0)
        self.uh.set_brightness(brightness)

    def clear(self):
        self.uh.clear()

    def show(self):
        self.uh.show()

    def set_pixel_rgb(self, x: int, y: int, r: int, g: int, b: int):
        r = min(max(int(r), 0), 255)
        g = min(max(int(g), 0), 255)
        b = min(max(int(b), 0), 255)
        if self.flip:
            y = self.height - y - 1
            x = self.width - x - 1
        self.uh.set_pixel(x, y, r, g, b)

    def set_button_handler(self, button: ButtonT, func: Callable[[], None]):
        self.buttons[button].when_pressed = func


class DisplayHatMini(Screen):
    """
    See: https://github.com/pimoroni/displayhatmini-python
    """

    def __init__(self, brightness: float = 0.25, flip: bool = False) -> None:

        width = _DisplayHATMini.WIDTH
        height = _DisplayHATMini.HEIGHT
        buffer = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(buffer)

        self.width = width
        self.height = height
        self.buff = buffer
        self.draw = draw
        self.flip = flip
        self.disp = _DisplayHATMini(self.buff)

        self.disp.set_led(0.05, 0.00, 0.00)

    def set_brightness(self, brightness: float):
        log.debug(f"Screen brightness set to {brightness:.2f}")
        brightness = min(max(brightness, 0.02), 1.0)
        # self.disp.set_led(brightness, brightness, brightness)

    def clear(self):
        self.disp.clear()

    def show(self):
        self.disp.display()

    def set_pixel_rgb(self, x: int, y: int, r: int, g: int, b: int):
        r = min(max(int(r), 0), 255) / 255.0
        g = min(max(int(g), 0), 255) / 255.0
        b = min(max(int(b), 0), 255) / 255.0
        if self.flip:
            y = self.height - y - 1
            x = self.width - x - 1
        self.buff.putpixel((x, y), (r, g, b))

    def set_button_handler(self, button: ButtonT, func: Callable[[], None]):
        # TODO: this
        pass
