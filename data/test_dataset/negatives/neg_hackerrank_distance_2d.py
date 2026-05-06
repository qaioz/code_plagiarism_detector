"""
Negative example. A 2D Point class with Euclidean distance. Same shape as
hackerrank/Class 2 - Find the Torsional Angle.py (a class with math) but in
2D, with a single distance method instead of dot/cross/absolute.
"""
import math


class Point2D:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)
