import datetime
import numpy as np

from abc import ABC, abstractmethod


__all__ = ["Animation"]


class Animation(ABC):
    """
    Buttons A/B are reserved for brightness control so are not handled by the animation.
    """

    @abstractmethod
    def step(self, time: datetime.datetime) -> np.ndarray:
        pass

    def handle_X(self):
        pass

    def handle_Y(self):
        pass
