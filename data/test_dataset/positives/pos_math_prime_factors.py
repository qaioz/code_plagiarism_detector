"""
Plagiarized from math/prime_factors.py.

Same strip-2s-then-trial-divide-by-odd-numbers approach. Renamed factors ->
primes, i -> divisor, kept the loop structure identical.
"""


def prime_factorization(n):
    primes = []
    while n % 2 == 0:
        primes.append(2)
        n //= 2
    divisor = 3
    while divisor * divisor <= n:
        while n % divisor == 0:
            primes.append(divisor)
            n //= divisor
        divisor += 2
    if n > 2:
        primes.append(n)
    return primes
