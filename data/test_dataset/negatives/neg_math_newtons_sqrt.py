"""
Negative example. Newton's method for square root via iterative refinement
(x_{n+1} = (x_n + n/x_n) / 2). Different problem from anything in math/ —
the corpus has polygon geometry, GCD/LCM, and prime-factorization
utilities, but no numerical root-finding.
"""


def newtons_sqrt(n, tolerance=1e-10, max_iter=100):
    if n < 0:
        raise ValueError("cannot compute sqrt of a negative number")
    if n == 0:
        return 0
    x = n / 2 if n >= 1 else 1.0
    for _ in range(max_iter):
        next_x = 0.5 * (x + n / x)
        if abs(next_x - x) < tolerance:
            return next_x
        x = next_x
    return x
