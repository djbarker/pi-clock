import numpy as np
import logging

from abc import ABC, abstractmethod
from colorsys import hsv_to_rgb
from gpiozero import Button
from unicornhatmini import UnicornHATMini
from typing import Callable, Literal

from utils import cmyk_to_rgb


__all__ = ["Screen", "UnicornHATMini"]

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


class UnicornHATMini(Screen):

    def __init__(self, brightness: float = 0.25, flip: bool = False):
        self.uh = UnicornHATMini()

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
