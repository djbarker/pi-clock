import argparse as ap
import datetime
import numpy as np
import time
import sys

from colorsys import hsv_to_rgb
from dataclasses import dataclass
from typing import Union, Literal

from PIL import Image, ImageDraw, ImageFont
from unicornhatmini import UnicornHATMini
from gpiozero import Button
from signal import pause


from font_v2 import chars

# To turn off the status LED do the following in bash:
#
# $ echo 0 | sudo tee /sys/class/leds/ACT/brightness
#
# See: https://forums.raspberrypi.com/viewtopic.php?t=326862

MAX_T = 1_000_000.0

MIN_BRIGHTNESS_RED = 4


def cmyk_to_rgb(
    c: float,
    m: float,
    y: float,
    k: float,
    cmyk_scale: float = 255,
    rgb_scale: float = 255,
):
    r = rgb_scale * (1.0 - c / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    g = rgb_scale * (1.0 - m / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    b = rgb_scale * (1.0 - y / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    return r, g, b


class Screen:

    def __init__(self, brightness: float = 0.25, flip: bool = False):
        self.uh = UnicornHATMini()

        # TODO: what does this do?
        rotation = 0
        self.uh.set_rotation(rotation)

        self.width, self.height = self.uh.get_shape()

        self.uh.set_brightness(brightness)
        self.flip = flip

    def set_brightness(self, brightness: float):
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


def blur(img: np.ndarray, kernel: np.ndarray):
    """
    Very dummy 2d convolution.
    Doesn't really make sense for HSV images.
    # TODO: This could definitely be done in a more efficient fashion.
    """
    assert kernel.shape == (3, 3)
    img_ = np.zeros_like(img, dtype=np.float32)
    for c in [0, 1, 2]:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                for i in [-1, 0, 1]:
                    x_ = x + i
                    for j in [-1, 0, 1]:
                        y_ = y + j
                        if (0 <= x_ < img.shape[0]) and (0 <= y_ < img.shape[1]):
                            img_[x, y, c] += kernel[i + 1, j + 1] * img[x_, y_, c]
    return img_


@dataclass
class SplatParams:
    splat_prob: float
    splat_kernel: Union[np.ndarray, None]
    hue_freq: float
    hue_width: float
    decay_frac: float

    def __post_init__(self):
        assert 0 <= self.splat_prob <= 1
        assert 0 <= self.decay_frac <= 1
        assert 0 <= self.hue_width <= 1
        assert 0 <= self.hue_freq


class GenericSplatAnimation:

    def __init__(
        self,
        width: int,
        height: int,
        params: SplatParams,
    ) -> None:
        self.width = width
        self.height = height

        self.set_params(params)

        self.img_rgb = np.zeros((width, height, 3))

    def set_params(self, params: SplatParams):
        self.prob = params.splat_prob
        self.kernel = params.splat_kernel
        self.hue_freq = params.hue_freq
        self.hue_width = params.hue_width
        self.decay_frac = params.decay_frac

        if self.kernel is not None:
            self.kernel = self.kernel / np.sum(self.kernel)

    def step(self, t: float) -> np.ndarray:
        self.img_rgb *= self.decay_frac

        if self.kernel is not None:
            self.img_rgb = blur(self.img_rgb, self.kernel)

        if np.random.uniform() < self.prob:
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)
            t = t * self.hue_freq
            h = np.random.uniform(t, t + self.hue_width)
            h = np.fmod(h, 1.0)
            rgb = np.array(hsv_to_rgb(h, 1.0, 1.0))
            self.img_rgb[x, y, :] = rgb * 255

        return self.img_rgb


class ChangingSplatAnim:

    def __init__(self, width: int, height: int, pattern: str):

        dots = SplatParams(0.6, None, 1 / 30, 0.7, 0.95)

        splat = SplatParams(
            0.5,
            np.array(
                [
                    [0.10, 1.00, 0.10],
                    [1.00, 7.00, 1.00],
                    [0.10, 1.00, 0.10],
                ]
            ),
            1 / 30,
            0.6,
            0.95,
        )

        matrix = SplatParams(
            0.5,
            np.array(
                [
                    [0.00, 1.00, 0.00],
                    [0.05, 5.00, 0.05],
                    [0.05, 5.00, 0.05],
                ]
            ),
            1 / 30,
            0.3,
            0.9,
        )

        self.params = {"dots": dots, "splat": splat, "matrix": matrix}
        self.anim = GenericSplatAnimation(width, height, self.params[pattern])
        self.swap_dt = datetime.timedelta(seconds=60.0)
        self.swap_t = datetime.datetime.now() + self.swap_dt

    def step(self, t: float) -> np.ndarray:
        now_t = datetime.datetime.now()
        if now_t >= self.swap_t:
            i = np.random.randint(len(self.params))
            k = list(self.params.values())[i]
            self.anim.set_params(k)
            self.swap_t += self.swap_dt

        return self.anim.step(t)


class SolidColourAnim:

    def __init__(self, width: int, height: int, r: float, g: float, b: float):
        self.img = np.zeros((width, height, 3))
        self.img[:, :, 0] = r
        self.img[:, :, 1] = g
        self.img[:, :, 2] = b

    def step(self, _t: float) -> np.ndarray:
        return self.img


mode = Literal["time", "ambience"]

COLOUR_MAP = {
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "yellow": (2, 1, 0),
    "cyan": (0, 1, 1),
    "magenta": (2, 0, 1),
}

COLOUR_NAMES = list(COLOUR_MAP.keys())


class TimeAnim:

    def __init__(self, width: int, height: int, background) -> None:
        self.w = width
        self.h = height
        self.background = background
        self.mode = "ambience" if self.is_day_time else "time"
        self.colour = "red"

    @property
    def is_day_time(self):
        time = datetime.datetime.now()
        day_time = time.hour >= 9 and time.hour < 22
        return day_time

    def cycle_mode(self):
        # better ways to do this, whatever
        if self.mode == "time":
            self.mode = "ambience"
        elif self.mode == "ambience":
            self.mode = "time"

        print(f"Set mode to {self.mode}")

    def cycle_colour(self):
        cidx = COLOUR_NAMES.index(self.colour)
        cidx = (cidx + 1) % len(COLOUR_NAMES)
        self.colour = COLOUR_NAMES[cidx]

        print(f"Set colour to {self.colour}")

    def step(self, t) -> np.ndarray:
        img = np.zeros((self.w, self.h, 3))

        time = datetime.datetime.now()
        dlim = ":" if (time.second % 2) == 0 else " "
        text = f"{time:%H}{dlim}{time:%M}"

        r, g, b = 0, 0, 0

        # if self.mode == "ambience" and self.background is not None:
        if self.is_day_time and self.background is not None:
            img = self.background.step(t)
        # elif self.mode == "time":
        else:
            if self.is_day_time:
                h = np.fmod(t, 300)
                r, g, b = np.array(hsv_to_rgb(h, 0.8, 1.0)) * 255
            else:
                r, g, b = np.array(COLOUR_MAP[self.colour]) * MIN_BRIGHTNESS_RED

        offx = 0
        for c in text:
            c = chars[c]
            img[offx : offx + c.shape[0], 1:6, 0] += c * r
            img[offx : offx + c.shape[0], 1:6, 1] += c * g
            img[offx : offx + c.shape[0], 1:6, 2] += c * b
            offx += c.shape[0] + 1

        return img


class Overlay:
    def __init__(self, anim1, w1, anim2, w2):
        self.anim1 = anim1
        self.anim2 = anim2
        self.w1 = w1
        self.w2 = w2

    def step(self, t: float) -> np.ndarray:
        return self.anim1.step(t) * self.w1 + self.anim2.step(t) * self.w2


class Transition:

    def __init__(self, w, h, anim1, anim2):
        self.anim1 = anim1
        self.anim2 = anim2
        self.x = 0
        self.w = w

        self.img_rgb = np.zeros((w, h, 3))

    def step(self, t: float) -> np.ndarray:
        img_1 = self.anim1.step(t)
        img_2 = self.anim2.step(t)

        self.img_rgb[: self.x, :, :] = img_2[: self.x, :, :]
        self.img_rgb[self.x :, :, :] = img_1[self.x :, :, :]
        self.img_rgb[self.x, :, :] = 10.0
        self.x += 1

        return self.img_rgb

    @property
    def done(self) -> bool:
        return self.x >= self.w


if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument("--brightness", "-b", type=float, default=0.1)
    parser.add_argument(
        "--pattern", "-p", choices={"dots", "matrix", "splat"}, default="matrix"
    )
    parser.add_argument("--flip", "-f", action="store_true")
    args = parser.parse_args()

    brightness = min(max(args.brightness, 0.02), 1.0)

    # setup screen & buttons

    screen = Screen(brightness=brightness, flip=args.flip)
    button_a = Button(5)
    button_b = Button(6)
    button_x = Button(16)
    button_y = Button(24)

    def brightness_change(incr: float):
        def _impl():
            global brightness
            brightness = min(max(brightness + incr, 0.02), 1.0)
            print(f"Set brightness to {brightness}")
            screen.set_brightness(brightness)

        return _impl

    button_a.when_pressed = brightness_change(0.05)
    button_b.when_pressed = brightness_change(-0.05)

    anim_ = ChangingSplatAnim(screen.width, screen.height, args.pattern)
    anim = TimeAnim(screen.width, screen.height, anim_)

    button_x.when_pressed = anim.cycle_mode
    button_y.when_pressed = anim.cycle_colour

    # main loop
    try:

        start_t = datetime.datetime.now()

        print(f"starting: {start_t:%F %H:%M:%S}")
        sys.stdout.flush()

        dt = 0.05
        t = 10
        while True:

            img_rgb = anim.step(t)
            screen.set_all_rgb(img_rgb)
            screen.show()
            time.sleep(dt)
            t += dt
            t = np.fmod(t, MAX_T)

    finally:
        print("Exiting ...")

        exit_anim = SolidColourAnim(screen.width, screen.height, 0, 0, 0)
        trans = Transition(screen.width, screen.height, anim, exit_anim)

        dt = 0.005
        while not trans.done:
            img_rgb = trans.step(t)
            screen.set_all_rgb(img_rgb)
            screen.show()
            time.sleep(dt)
            t += dt
            t = np.fmod(t, MAX_T)

        screen.clear()
        screen.show()
