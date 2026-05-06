"""
Negative example. A range-sum segment tree (recursive, 4n storage). Different
data structure than pyrival/FenwickTree.py — both answer prefix-sum-style
queries but the internal layout and logic are unrelated.
"""


class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self._build(arr, 0, 0, self.n - 1)

    def _build(self, arr, node, lo, hi):
        if lo == hi:
            self.tree[node] = arr[lo]
            return
        mid = (lo + hi) // 2
        self._build(arr, 2 * node + 1, lo, mid)
        self._build(arr, 2 * node + 2, mid + 1, hi)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def query(self, lo, hi):
        return self._query(0, 0, self.n - 1, lo, hi)

    def _query(self, node, nlo, nhi, qlo, qhi):
        if qhi < nlo or nhi < qlo:
            return 0
        if qlo <= nlo and nhi <= qhi:
            return self.tree[node]
        mid = (nlo + nhi) // 2
        return (self._query(2 * node + 1, nlo, mid, qlo, qhi)
                + self._query(2 * node + 2, mid + 1, nhi, qlo, qhi))
