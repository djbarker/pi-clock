import datetime
import logging
import numpy as np
import re
import requests

from dataclasses import dataclass

from anims import Animation

log = logging.getLogger(__name__)

# see: https://content.tfl.gov.uk/tfl-colour-standard-issue-08.pdf
# red appears less bright so I've down-weighted blue and green manually
LINE_COLORS = {
    "victoria": (0, 160, 223),
    "london-overground": (238, 60, 10),
    "jubilee": (80, 70, 72),
    "central": (225, 4, 4),
    "piccadilly": (0, 15, 159),
    "northern": (0, 0, 0),
    "elizabeth": (145, 42, 135),
}

BLOCK_SIZE_X_LINE = 3
BLOCK_SIZE_X_STAT = 2
BLOCK_SIZE_Y = 3


@dataclass
class Status:
    value: str

    @property
    def color(self) -> np.ndarray:
        if self.value.lower() == "good service":
            return np.array([0, 120, 0], dtype=np.float64)
        elif self.value.lower() == "minor delays":
            return np.array([130, 70, 0], dtype=np.float64)
        elif self.value.lower() == "unknown":
            return np.array([30, 0, 30], dtype=np.float64)
        else:
            return np.array([120, 0, 0], dtype=np.float64)

    @property
    def is_good_service(self) -> bool:
        return self.value.lower() == "good service"

    @property
    def is_error(self) -> bool:
        return self.value.lower() == "unknown"


class TubeStatusAnim(Animation):
    """
    Display a very basic tube status indicator with simple blocks of color.
    """

    def __init__(
        self,
        width: int,
        height: int,
    ):
        self.width = width
        self.height = height

        self.update_period = datetime.timedelta(seconds=300)
        self.next_update = datetime.datetime.now()
        self.lines = [
            "victoria",
            "london-overground",
            "central",
            "jubilee",
            "elizabeth",
            "piccadilly",
        ]
        self.statuses = [Status("unknown") for l in self.lines]

    def step(self, time: datetime.datetime) -> np.ndarray:

        # first get statuses if needed

        if time >= self.next_update:  # or any(s.is_error for s in self.statuses):

            statuses = []
            for line in self.lines:
                try:
                    response = requests.get(f"http://api.tfl.gov.uk/Line/{line}/Status")

                    if response.status_code != 200:
                        statuses.append(Status("error"))
                        log.warning(
                            f"response: {response.text} [line={line}, code={response.status_code}]"
                        )
                        continue

                    body = response.content.decode("ascii")
                    match = re.search(r'"statusSeverityDescription":"(.*?)"', body)
                    statuses.append(Status(match.groups()[0]))
                except:
                    statuses.append(Status("unknown"))

            self.statuses = statuses
            self.next_update = time + self.update_period

        # then plot image

        img = np.zeros((self.width, self.height, 3))
        off_x = 0
        off_y = 0
        for line, status in zip(self.lines, self.statuses):
            col_l = np.array(LINE_COLORS[line]) * 0.7
            col_s = status.color

            # blink degraded service statuses
            if not status.is_good_service and (time.second % 2 == 0):
                col_s *= 0.5

            # fmt: off
            img[off_x + 0 * BLOCK_SIZE_X_LINE : off_x + BLOCK_SIZE_X_LINE + 0 * BLOCK_SIZE_X_STAT, off_y : off_y + BLOCK_SIZE_Y, :] = col_l
            img[off_x + 1 * BLOCK_SIZE_X_LINE : off_x + BLOCK_SIZE_X_LINE + 1 * BLOCK_SIZE_X_STAT, off_y : off_y + BLOCK_SIZE_Y, :] = col_s
            # fmt: on

            off_y += BLOCK_SIZE_Y + 1
            if off_y + BLOCK_SIZE_Y > self.height:
                off_y = 0
                off_x += BLOCK_SIZE_X_LINE + BLOCK_SIZE_X_STAT + 1

                if off_x + BLOCK_SIZE_X_LINE + BLOCK_SIZE_X_STAT > self.width:
                    break

        return img
