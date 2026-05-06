"""
Negative example. Union-Find without path compression and without union-by-
size — naive recursive find, parent-pointer union. Same data structure as
pyrival/DisjointSetUnion.py but with materially different logic and worse
asymptotics.
"""


class NaiveUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] == x:
            return x
        return self.find(self.parent[x])

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry
