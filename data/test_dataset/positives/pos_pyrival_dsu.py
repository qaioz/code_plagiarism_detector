"""
Plagiarized from pyrival/DisjointSetUnion.py.

Same union-find with path compression and union-by-size. Renamed parent ->
leaders, size -> cluster_size, parameter names a/b -> x/y. Kept the
num_sets counter and __len__ method.
"""


class DSU:
    def __init__(self, n):
        self.leaders = list(range(n))
        self.cluster_size = [1] * n
        self.num_sets = n

    def find(self, x):
        start = x
        while x != self.leaders[x]:
            x = self.leaders[x]
        while start != x:
            self.leaders[start], start = x, self.leaders[start]
        return x

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y:
            if self.cluster_size[x] < self.cluster_size[y]:
                x, y = y, x
            self.num_sets -= 1
            self.leaders[y] = x
            self.cluster_size[x] += self.cluster_size[y]

    def set_size(self, x):
        return self.cluster_size[self.find(x)]

    def __len__(self):
        return self.num_sets
