"""
Plagiarized from pyrival/Fraction.py.

Same Fraction class normalized by gcd in __init__, with the same
cross-multiply semantics for add/sub/mul/div and the same comparison
semantics. Reformatted from one-line lambdas to def methods, renamed
num/den -> numerator/denominator. Kept the standalone gcd helper.
"""


def gcd(x, y):
    while y:
        x, y = y, x % y
    return x


class Fraction:
    def __init__(self, numerator=0, denominator=1):
        g = gcd(numerator, denominator)
        self.numerator = numerator // g
        self.denominator = denominator // g

    def __add__(self, other):
        return Fraction(
            self.numerator * other.denominator + other.numerator * self.denominator,
            self.denominator * other.denominator,
        )

    def __sub__(self, other):
        return Fraction(
            self.numerator * other.denominator - other.numerator * self.denominator,
            self.denominator * other.denominator,
        )

    def __mul__(self, other):
        return Fraction(
            self.numerator * other.numerator,
            self.denominator * other.denominator,
        )

    def __truediv__(self, other):
        return Fraction(
            self.numerator * other.denominator,
            self.denominator * other.numerator,
        )

    def __neg__(self):
        return Fraction(-self.numerator, self.denominator)

    def __eq__(self, other):
        return self.numerator * other.denominator == other.numerator * self.denominator

    def __lt__(self, other):
        return self.numerator * other.denominator < other.numerator * self.denominator
