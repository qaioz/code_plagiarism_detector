"""
Plagiarized from keon_graph/bellman_ford.py.

Same Bellman-Ford relaxation loop and negative-cycle detection pass. Renamed
distance/predecessor -> dist/prev, current_node/neighbor -> u/v, edge_weight
-> w. Inlined initialization instead of a helper.
"""


def bellman_ford(graph, source):
    dist = {}
    prev = {}
    nodes = set(graph.keys())
    for nbrs in graph.values():
        nodes.update(nbrs.keys())
    for n in nodes:
        dist[n] = float('inf')
        prev[n] = None
    dist[source] = 0

    for _ in range(1, len(graph)):
        for u in graph:
            for v, w in graph[u].items():
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    prev[v] = u

    for u in graph:
        for v, w in graph[u].items():
            if dist[v] > dist[u] + w:
                return False
    return True
