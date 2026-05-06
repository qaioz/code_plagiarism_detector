"""
Plagiarized from keon_graph/all_pairs_shortest_path.py.

Same Floyd-Warshall triple-nested relaxation with the via-vertex k as the
outer index. Renamed adjacency_matrix -> dist_matrix, new_array -> dist;
swapped copy.deepcopy for a list-comprehension shallow copy of each row
(rows hold floats so it's equivalent). Dropped type annotations + docstring.
"""


def floyd_warshall(dist_matrix):
    dist = [row[:] for row in dist_matrix]
    n = len(dist)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
