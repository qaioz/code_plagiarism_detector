"""
Plagiarized from keon_graph/a_star.py.

Same heap-driven A* with f = g + h, visited set, and path-tracking via list
copies. Renamed a_star -> astar_search, h -> heuristic, open_set -> frontier.
Dropped the type annotations and docstring; same control flow otherwise.
"""
import heapq


def astar_search(graph, start, goal, heuristic):
    frontier = []
    heapq.heappush(frontier, (heuristic(start), 0, start, [start]))
    seen = set()

    while frontier:
        f, g, current, path = heapq.heappop(frontier)
        if current == goal:
            return path, g
        if current in seen:
            continue
        seen.add(current)
        for nbr, cost in graph.get(current, []):
            if nbr in seen:
                continue
            new_g = g + cost
            new_f = new_g + heuristic(nbr)
            heapq.heappush(frontier, (new_f, new_g, nbr, path + [nbr]))

    return None, float('inf')
