import datetime
import numpy as np

from anims import Animation


class SolidColorAnim(Animation):

    def __init__(self, width: int, height: int, r: float, g: float, b: float):
        self.img = np.zeros((width, height, 3))
        self.img[:, :, 0] = r
        self.img[:, :, 1] = g
        self.img[:, :, 2] = b

    def step(self, _t: datetime.datetime) -> np.ndarray:
        return self.img
