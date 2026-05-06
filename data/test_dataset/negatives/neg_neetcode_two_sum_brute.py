"""
Negative example. Solves Two Sum but with the O(n^2) brute-force nested loop —
fundamentally different algorithm than the hash-map approach in
neetcode/0001-two-sum.py.
"""


def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
