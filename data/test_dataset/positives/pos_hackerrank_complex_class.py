"""
Plagiarized from hackerrank/Classes: Dealing with Complex Numbers.py.

Same Complex class with __add__, __sub__, __mul__, __truediv__ and mod.
Renamed Complex -> ComplexNumber, real/imaginary -> re/im, parameter no -> other.
Dropped the __str__ formatting block (out of scope for the arithmetic test).
"""
import math


class ComplexNumber:
    def __init__(self, re, im):
        self.re = re
        self.im = im

    def __add__(self, other):
        return ComplexNumber(self.re + other.re, self.im + other.im)

    def __sub__(self, other):
        return ComplexNumber(self.re - other.re, self.im - other.im)

    def __mul__(self, other):
        return ComplexNumber(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    def __truediv__(self, other):
        denom = float(other.re ** 2 + other.im ** 2)
        prod = self * ComplexNumber(other.re, -other.im)
        return ComplexNumber(prod.re / denom, prod.im / denom)

    def mod(self):
        return ComplexNumber(math.pow(self.re ** 2 + self.im ** 2, 0.5), 0)
