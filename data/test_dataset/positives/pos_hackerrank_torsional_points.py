"""
Plagiarized from hackerrank/Class 2 - Find the Torsional Angle.py.

Same 3D Points class with __sub__, dot, cross, magnitude. Renamed Points ->
Vector3D, parameter 'no' -> 'other', renamed local variables (x_s, y_s, z_s
-> dx, dy, dz). Kept all four methods.
"""


class Vector3D:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __sub__(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return Vector3D(dx, dy, dz)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        cx = self.y * other.z - self.z * other.y
        cy = self.x * other.z - self.z * other.x
        cz = self.x * other.y - self.y * other.x
        return Vector3D(cx, cy, cz)

    def absolute(self):
        return pow((self.x ** 2 + self.y ** 2 + self.z ** 2), 0.5)
