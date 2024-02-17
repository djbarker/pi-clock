import argparse as ap
import datetime
import logging
import numpy as np
import queue
import time
import sys

from colorsys import hsv_to_rgb
from dataclasses import dataclass
from typing import Callable, Union, Literal

from PIL import Image, ImageDraw, ImageFont
from gpiozero import Button
from signal import pause

from screen import Screen
from anims import Animation
from anims.time import TimeAnim, is_daytime, get_next_time_of_day
from anims.splat import ChangingSplatAnim
from anims.tube import TubeStatusAnim

# To turn off the status LED do the following in bash:
#
# $ echo 0 | sudo tee /sys/class/leds/ACT/brightness
#
# See: https://forums.raspberrypi.com/viewtopic.php?t=326862

log = logging.getLogger(__name__)


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


@dataclass
class Event:
    time: datetime.time
    func: Callable[[datetime.time], None]

    def __lt__(self, event: "Event") -> bool:
        return self.time < event.time

    def __gt__(self, event: "Event") -> bool:
        return self.time > event.time

    def __le__(self, event: "Event") -> bool:
        return self.time <= event.time

    def __ge__(self, event: "Event") -> bool:
        return self.time >= event.time


class MainLoop:
    """
    Handles transitioning between animations and telling the current animation to handle the button presses.
    """

    DAYTIME = datetime.time(8, 30)
    NIGHTTIME = datetime.time(22, 30)

    ANIM_PERIOD = datetime.timedelta(seconds=45)

    def __init__(self, brightness: float, flip: bool):

        # setup screen & buttons

        self.screen = Screen(brightness=brightness, flip=flip)
        self.button_a = Button(5)
        self.button_b = Button(6)
        self.button_x = Button(16)
        self.button_y = Button(24)

        def brightness_change(incr: float):
            def _impl():
                global brightness
                brightness = min(max(brightness + incr, 0.02), 1.0)
                log.info(f"Set brightness to {brightness}")
                self.screen.set_brightness(brightness)

            return _impl

        self.button_a.when_pressed = brightness_change(0.05)
        self.button_b.when_pressed = brightness_change(-0.05)

        self.button_x.when_pressed = self.handle_X
        self.button_y.when_pressed = self.handle_Y

        # set up the animations

        w, h = self.screen.width, self.screen.height

        self.anims = {
            "time": TimeAnim(w, h, MainLoop.DAYTIME, MainLoop.NIGHTTIME),
            "tube": TubeStatusAnim(w, h),
            "splat": ChangingSplatAnim(w, h, "matrix"),
        }

        self.anim_idx = 0  # start with time
        self.anim_keys = list(self.anims.keys())

        # set up next transition
        self.events = queue.PriorityQueue()

        now = datetime.datetime.now()

        if self.is_daytime(now):
            if time := self.incr_time(now, self.ANIM_PERIOD):
                self.schedule(
                    time,
                    lambda time: self.cycle_anim(time),
                )

        def _on_daytime(time: datetime.datetime):
            self.screen.set_brightness(0.3)
            self.cycle_anim(time)

            # schedule tomorrow's daytime
            self.schedule(get_next_time_of_day(time, self.NIGHTTIME), _on_nighttime)

        def _on_nighttime(time: datetime.datetime):
            self.screen.set_brightness(0.05)
            self.cycle_anim(time, "time")

            # schedule tomorrow's nighttime
            self.schedule(get_next_time_of_day(time, self.NIGHTTIME), _on_nighttime)

        self.schedule(get_next_time_of_day(now, self.DAYTIME), _on_daytime)
        self.schedule(get_next_time_of_day(now, self.NIGHTTIME), _on_nighttime)

    def schedule(
        self, time: datetime.datetime, func: Callable[[datetime.datetime], None]
    ):
        log.debug(f"Scheduling event at {time:%F %H:%M:%S}")
        self.events.put(Event(time, func))

    def is_daytime(self, time: datetime.datetime) -> bool:
        return is_daytime(time, self.DAYTIME, self.NIGHTTIME)

    @staticmethod
    def incr_time(
        time: datetime.datetime, delta: datetime.timedelta
    ) -> Union[datetime.datetime, None]:
        """
        Increment the time but return none if we would cross a day/night- or night/day-time transition.
        """
        assert delta >= datetime.timedelta(
            seconds=0
        ), f"Cannot incr with negative delta! [{delta=}]"
        next_daytime = get_next_time_of_day(time, MainLoop.DAYTIME)
        next_nighttime = get_next_time_of_day(time, MainLoop.NIGHTTIME)
        time = time + delta
        if (time >= next_daytime) or (time >= next_nighttime):
            return None
        else:
            return time

    @property
    def curr_anim(self) -> Animation:
        return self.anims[self.anim_keys[self.anim_idx]]

    def cycle_anim(self, time: datetime.datetime, anim: Union[str, None] = None):
        if anim is None:
            self.anim_idx = (self.anim_idx + 1) % len(self.anim_keys)
        else:
            self.anim_idx = self.anim_keys.index(anim)

        if self.is_daytime(time) and (time_ := self.incr_time(time, self.ANIM_PERIOD)):
            self.schedule(time_, self.cycle_anim)

        log.debug(f"Set anim to {self.anim_keys[self.anim_idx]!r}")

    def step(self, time: datetime.datetime):

        while self.events:
            event = self.events.get()

            if event.time > time:
                # we've processed all events <= time
                self.events.put(event)
                break
            else:
                event.func(time)

        img_rgb = self.curr_anim.step(time)
        self.screen.set_all_rgb(img_rgb)
        self.screen.show()

    def clear(self):
        self.screen.clear()
        self.screen.show()

    def handle_X(self):
        self.curr_anim.handle_X()

    def handle_Y(self):
        self.curr_anim.handle_Y()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} - [{module}] {levelname} - {message}",
        style=r"{",
    )

    parser = ap.ArgumentParser()
    parser.add_argument("--brightness", "-b", type=float, default=0.1)
    parser.add_argument("--flip", "-f", action="store_true")
    args = parser.parse_args()

    # main loop

    log.info("Initializing")
    loop = MainLoop(args.brightness, args.flip)

    try:
        start_t = datetime.datetime.now()
        delta_t = 0.05

        log.info(f"Starting loop")

        while True:
            loop.step(datetime.datetime.now())
            time.sleep(delta_t)

            sys.stderr.flush()

    finally:
        log.info("Exiting")
        loop.clear()
