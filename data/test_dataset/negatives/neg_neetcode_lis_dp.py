"""
Negative example. Longest Increasing Subsequence via O(n^2) dynamic
programming — dp[i] = max(dp[j] + 1) for all j < i with nums[j] < nums[i].
Unrelated to any indexed neetcode problem (corpus has Two Sum, palindrome
checks, reverse-integer, etc., but no subsequence DP).
"""


def length_of_lis(nums):
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
