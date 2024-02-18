import datetime
import logging
import numpy as np

from colorsys import hsv_to_rgb
from typing import Literal, Union

from anims import Animation
from utils import MIN_BRIGHTNESS_R, MIN_BRIGHTNESS_G, MIN_BRIGHTNESS_B
from font_v2 import chars

__all__ = [
    "is_daytime",
    "is_nighttime",
    "get_next_time_of_day",
    "get_prev_time_of_day",
    "TimeAnim",
]

log = logging.getLogger(__name__)


COLOR_MAP = {
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "yellow": (2, 1, 0),
    "cyan": (0, 1, 1),
    "magenta": (2, 0, 1),
}

COLOR_NAMES = list(COLOR_MAP.keys())


def is_daytime(
    time: datetime.datetime, daytime: datetime.time, nighttime: datetime.time
) -> bool:
    time = time.time()
    return (daytime <= time) and (time < nighttime)


def is_nighttime(
    time: datetime.datetime, daytime: datetime.time, nighttime: datetime.time
) -> bool:
    return not is_daytime(time, daytime, nighttime)


def get_next_time_of_day(
    timestamp: datetime.datetime, time_of_day: datetime.time
) -> datetime.date:
    if timestamp.time() < time_of_day:
        today = timestamp.date()
        return datetime.datetime.combine(today, time_of_day)
    else:
        tomorrow = (timestamp + datetime.timedelta(days=1)).date()
        return datetime.datetime.combine(tomorrow, time_of_day)


def get_prev_time_of_day(
    timestamp: datetime.datetime, time_of_day: datetime.time
) -> datetime.date:
    if timestamp.time() >= time_of_day:
        today = timestamp.date()
        return datetime.datetime.combine(today, time_of_day)
    else:
        tomorrow = (timestamp - datetime.timedelta(days=1)).date()
        return datetime.datetime.combine(tomorrow, time_of_day)


class TimeAnim(Animation):

    def __init__(
        self, width: int, height: int, daytime: datetime.time, nighttime: datetime.time
    ) -> None:
        self.w = width
        self.h = height
        self.daytime = daytime
        self.nighttime = nighttime
        self.color = "red"

    def cycle_color(self):
        cidx = COLOR_NAMES.index(self.color)
        cidx = (cidx + 1) % len(COLOR_NAMES)
        self.color = COLOR_NAMES[cidx]

        log.info(f"Set color to {self.color}")

    def step(self, time: datetime.datetime) -> np.ndarray:

        dlim = ":" if (time.second % 2) == 0 else " "
        text = f"{time:%H}{dlim}{time:%M}"

        img = np.zeros((self.w, self.h, 3))
        r, g, b = 0, 0, 0

        if is_daytime(time, self.daytime, self.nighttime):
            hue_freq = 200.0
            h = np.fmod(time.timestamp(), hue_freq) / hue_freq
            r, g, b = np.array(hsv_to_rgb(h, 0.8, 1.0)) * 100
        else:
            r, g, b = np.array(COLOR_MAP[self.color])
            r *= MIN_BRIGHTNESS_R
            g *= MIN_BRIGHTNESS_G
            b *= MIN_BRIGHTNESS_B

        offx = 0
        for c in text:
            c = chars[c]
            img[offx : offx + c.shape[0], 1:6, 0] += c * r
            img[offx : offx + c.shape[0], 1:6, 1] += c * g
            img[offx : offx + c.shape[0], 1:6, 2] += c * b
            offx += c.shape[0] + 1

        return img

    def handle_X(self):
        self.cycle_color()
