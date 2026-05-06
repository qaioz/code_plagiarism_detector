"""
Plagiarized from math/calculate_area_of_polygon.py.

Same Shoelace formula with cumulative trapezoids. Renamed prev/curr ->
previous/point, dropped the unused n variable.
"""


def polygon_area(vertices):
    previous = vertices[-1]
    accum = 0
    for point in vertices:
        accum += (previous[0] + point[0]) * (previous[1] - point[1])
        previous = point
    return abs(accum / 2)
