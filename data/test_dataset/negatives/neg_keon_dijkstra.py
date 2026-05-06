"""
Negative example. Dijkstra shortest-path with a binary heap. Same problem
family as keon_graph/bellman_ford.py (single-source shortest path) but a
fundamentally different algorithm — assumes non-negative weights, uses a
priority queue.
"""
import heapq


def dijkstra(graph, source):
    dist = {n: float('inf') for n in graph}
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
