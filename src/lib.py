import numpy as np

from abc import ABC, abstractmethod

class Animation(ABC):

    @abstractmethod
    def step(self) -> np.ndarray:
        pass


