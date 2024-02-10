__all__ = [
    "MIN_BRIGHTNESS_R",
    "MIN_BRIGHTNESS_G",
    "MIN_BRIGHTNESS_B",
    "cmyk_to_rgb",
]

# Empirically the LEDs switch off below 4, but even at 4 green & blue are much more bright than red.
MIN_BRIGHTNESS_R = 6
MIN_BRIGHTNESS_G = 4
MIN_BRIGHTNESS_B = 4


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
