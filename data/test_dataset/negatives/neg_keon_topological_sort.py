"""
Negative example. Kahn's topological sort using in-degree counts and a
queue. Unrelated to any of the graph algorithms in keon_graph/.
"""
from collections import deque


def topological_sort(graph):
    indeg = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            indeg[v] = indeg.get(v, 0) + 1
            indeg.setdefault(u, 0)
    q = deque([u for u, d in indeg.items() if d == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == len(indeg) else []
