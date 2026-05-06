"""
Negative example. Sieve of Eratosthenes — generates *all* primes up to n by
marking multiples of each prime. Same domain as math/prime_factors.py
(primality) but a fundamentally different algorithm: prime_factors does
trial-division to factorize a single number; this enumerates primes in a
range. Different signature, different control flow, different output type.
"""


def sieve_of_eratosthenes(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    p = 2
    while p * p <= n:
        if is_prime[p]:
            for multiple in range(p * p, n + 1, p):
                is_prime[multiple] = False
        p += 1
    return [i for i, prime in enumerate(is_prime) if prime]
