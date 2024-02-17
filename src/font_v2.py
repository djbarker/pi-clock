import numpy as np

from textwrap import dedent

# Based on  https://www.fontspace.com/teeny-tiny-pixls-font-f30095
# For some reason the ttf downloaded does not work with PIL

_0 = """\
###
#.#
#.#
#.#
###
"""

_1 = """\
.#.
##.
.#.
.#.
###
"""

_2 = """\
###
..#
###
#..
###
"""

_3 = """\
###
..#
###
..#
###
"""

_4 = """\
#.#
#.#
###
..#
..#
"""

_5 = """\
###
#..
###
..#
###
"""

_6 = """\
###
#..
###
#.#
###
"""

_7 = """\
###
#.#
..#
.#.
.#.
"""

_8 = """
###
#.#
###
#.#
###
"""

_9 = """\
###
#.#
###
..#
..#
"""

_colon = """
.
#
.
#
.
"""

_space = """
.
.
.
.
.
"""

chars = {
    "0": _0,
    "1": _1,
    "2": _2,
    "3": _3,
    "4": _4,
    "5": _5,
    "6": _6,
    "7": _7,
    "8": _8,
    "9": _9,
    ":": _colon,
    " ": _space,
}


def _to_numpy(c: str) -> np.ndarray:
    c = dedent(c.strip())
    w = list(c).index("\n")
    a = np.zeros((w, 5))
    i = 0
    j = 0
    for cc in c:
        if cc == "\n":
            i = 0
            j += 1
        else:
            a[i, j] = 1 if cc == "#" else 0
            i += 1
    return a


for k, v in chars.items():
    chars[k] = _to_numpy(v)
