"""
Plagiarized from keon_graph/check_bipartite.py.

Same BFS two-colouring approach with the same self-loop short-circuit and
same conflict check. Renamed adj_list -> matrix, set_type -> color,
current/adjacent -> u/v.
"""
from collections import deque


def is_bipartite(matrix):
    n = len(matrix)
    color = [-1] * n
    color[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        if matrix[u][u]:
            return False
        for v in range(n):
            if matrix[u][v]:
                if color[v] == color[u]:
                    return False
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    q.append(v)
    return True
