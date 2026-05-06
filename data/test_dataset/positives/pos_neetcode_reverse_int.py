"""
Plagiarized from neetcode/0007-reverse-integer.py.

Same overflow-checked digit-by-digit reversal. Inlined the MIN/MAX constants,
renamed res -> out, flattened the class to a function.
"""
import math


def reverse_integer(x):
    out = 0
    while x:
        d = int(math.fmod(x, 10))
        x = int(x / 10)
        if out > 214748364 or (out == 214748364 and d > 7):
            return 0
        if out < -214748364 or (out == -214748364 and d < -8):
            return 0
        out = out * 10 + d
    return out
