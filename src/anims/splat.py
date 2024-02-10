import datetime
import numpy as np

from colorsys import hsv_to_rgb
from dataclasses import dataclass
from typing import Union

from anims import Animation

__all__ = [
    "SplatParams",
    "GenericSplatAnimation",
    "ChangingSplatAnimation",
]


MAX_T = 1_000_000.0


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


class GenericSplatAnimation(Animation):

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
            t = np.fmod(t.timestamp(), MAX_T) * self.hue_freq
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

    def swap_anim(self, t: datetime.datetime):
        i = np.random.randint(len(self.params))
        k = list(self.params.values())[i]
        self.anim.set_params(k)
        self.swap_t = t + self.swap_dt

    def step(self, t: datetime.datetime) -> np.ndarray:
        if t >= self.swap_t:
            self.swap_anim(t)

        return self.anim.step(t)

    def handle_X(self):
        self.swap_anim(datetime.datetime.now())
