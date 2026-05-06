"""
Negative example. Kruskal's minimum spanning tree using sorted edges and
inline path-compressed union-find. Different problem from any indexed
keon_graph algorithm — corpus has shortest-path (Bellman-Ford, A*,
Floyd-Warshall) and bipartite-check, but no MST. The union-find is wrapped
in MST-construction logic, not exposed as a standalone DSU.
"""


def kruskals_mst(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True

    edges_sorted = sorted(edges, key=lambda e: e[2])
    mst = []
    for u, v, w in edges_sorted:
        if union(u, v):
            mst.append((u, v, w))
            if len(mst) == n - 1:
                break
    return mst
