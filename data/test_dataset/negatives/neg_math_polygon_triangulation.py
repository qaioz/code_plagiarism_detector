"""
Negative example. Computes polygon area by fan-triangulation from the first
vertex. Solves the same problem as math/calculate_area_of_polygon.py but with
a different algorithm (sum of triangle areas instead of the Shoelace formula).
"""


def triangulated_polygon_area(verts):
    if len(verts) < 3:
        return 0
    total = 0
    x0, y0 = verts[0]
    for i in range(1, len(verts) - 1):
        x1, y1 = verts[i]
        x2, y2 = verts[i + 1]
        total += abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2
    return total
