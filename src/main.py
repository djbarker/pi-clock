import argparse as ap
import datetime
import logging
import numpy as np
import os
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
from anims.time import TimeAnim, is_daytime, get_prev_time_of_day, get_next_time_of_day
from anims.splat import ChangingSplatAnim
from anims.tube import TubeStatusAnim

# To turn off the status LED do the following in bash:
#
# $ echo 0 | sudo tee /sys/class/leds/ACT/brightness
#
# See: https://forums.raspberrypi.com/viewtopic.php?t=326862

log = logging.getLogger(__name__)

# requests is too verbose
logging.getLogger("requests").setLevel(logging.WARN)


@dataclass
class Event:
    """
    Run the given callable at (or after) the specified time.
    """

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


def peek(pq: queue.PriorityQueue) -> Event:
    temp = pq.get()
    pq.put(temp)
    return temp


class MainLoop:
    """
    Handles transitioning between animations and telling the current animation to handle the button presses.
    """

    DAYTIME = datetime.time(8, 0)
    NIGHTTIME = datetime.time(22, 0)

    ANIM_PERIOD = datetime.timedelta(seconds=45)

    # Given ANIM_PERIOD of 45 sec expect no more than O(2000) in a 24 hour period
    MAX_EVENTS = 10_000

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

        # start schedule from previous event to ensure current state is correct
        now = datetime.datetime.now()
        self.schedule(get_prev_time_of_day(now, self.DAYTIME), self.schedule_day)

    def schedule_day(self, time: datetime.datetime):

        # schedule all animation transitions for today
        count = 0
        time_ = time
        while time_ := self.incr_time(time_, self.ANIM_PERIOD):
            self.schedule(time_, lambda _: self.cycle_anim)
            count += 1

        log.info(f"Scheduled {count:,d} transitions for {time.date():%F}.")

        # just show time at night
        self.schedule(
            get_next_time_of_day(time, self.NIGHTTIME), lambda _: self.set_anim("time")
        )

        # schedule brightness transitions
        self.screen.set_brightness(0.05)
        for t, b in [
            (datetime.time(9, 0), 0.10),
            (datetime.time(9, 30), 0.15),
            (datetime.time(10, 0), 0.20),
            (datetime.time(21, 0), 0.15),
            (datetime.time(21, 30), 0.10),
            (datetime.time(23, 30), 0.05),
        ]:
            self.schedule(
                get_next_time_of_day(time, t),
                lambda _t: self.screen.set_brightness(b),
            )

        # schedule tomorrow
        time_ = get_next_time_of_day(time, self.DAYTIME)
        self.schedule(time_, self.schedule_day)
        log.info(f"Will schedule tomorrow at {time_:%F %H:%M:%S}.")

    def schedule(
        self, time: datetime.datetime, func: Callable[[datetime.datetime], None]
    ):
        log.debug(f"Scheduling event at {time:%F %H:%M:%S}")
        self.events.put(Event(time, func))

        if self.events.qsize() > self.MAX_EVENTS:
            raise ValueError(f"Too many scheduled events!")

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

    def cycle_anim(self):
        self.anim_idx = (self.anim_idx + 1) % len(self.anim_keys)

    def set_anim(self, anim_name: str):
        self.anim_idx = self.anim_keys.index(anim_name)
        assert self.anim_idx > 0, f"Unknown anim_name! [{anim_name!r}]"

    def step(self, time: datetime.datetime):

        while self.events:
            if peek(self.events).time > time:
                # we've processed all events <= time
                break
            else:
                self.events.get().func(time)

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
        level=logging.getLevelName(os.environ.pop("LOG_LEVEL", logging.INFO)),
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
