"""
Plagiarized from pyrival/FenwickTree.py.

Same BIT (binary indexed tree) with the same in-place transform from list,
same point-update + prefix-sum-query, same findkth bit-trick. Renamed bit
-> tree, idx -> i in update, end -> stop in query.
"""


class BIT:
    def __init__(self, x):
        self.tree = x
        for i in range(len(x)):
            j = i | (i + 1)
            if j < len(x):
                x[j] += x[i]

    def update(self, i, x):
        while i < len(self.tree):
            self.tree[i] += x
            i |= i + 1

    def query(self, stop):
        x = 0
        while stop:
            x += self.tree[stop - 1]
            stop &= stop - 1
        return x

    def findkth(self, k):
        i = -1
        for d in reversed(range(len(self.tree).bit_length())):
            right = i + (1 << d)
            if right < len(self.tree) and k >= self.tree[right]:
                i = right
                k -= self.tree[i]
        return i + 1
