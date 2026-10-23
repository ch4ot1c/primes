import math
import os
import time
import webbrowser
from bisect import bisect_right
from collections import Counter

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import quad

# ==============================================================================
# SECTION 1: SIEVE AND NUMBER THEORY IMPLEMENTATIONS
# ==============================================================================

def sieve_of_eratosthenes(n: int) -> list[int]:
    """
    Implements the Sieve of Eratosthenes to find all prime numbers up to n.

    This is the classical sieve algorithm. It works by creating a boolean array
    representing the numbers from 0 to n. It then iteratively marks as
    composite (i.e., not prime) the multiples of each prime, starting with 2.

    Complexity:
    - Time: O(n log log n)
    - Space: O(n)

    Args:
        n: The upper limit for the sieve (inclusive).

    Returns:
        A list of all prime numbers less than or equal to n.
    """
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.sqrt(n)) + 1):
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
    return [p for p, is_p in enumerate(is_prime) if is_p]

def sieve_of_atkin(n: int) -> list[int]:
    """
    Implements the Sieve of Atkin to find all prime numbers up to n.

    The Sieve of Atkin is a more modern algorithm that identifies numbers as prime
    if they have an odd number of representations by certain quadratic forms.

    Complexity:
    - Time: O(n / log log n)
    - Space: O(n)

    Args:
        n: The upper limit for the sieve (inclusive).

    Returns:
        A list of all prime numbers less than or equal to n.
    """
    if n < 2:
        return []
    primes = [p for p in [2, 3] if p <= n]
    sieve = [False] * (n + 1)
    x = 1
    while x * x <= n:
        y = 1
        while y * y <= n:
            k = 4 * x * x + y * y
            if k <= n and (k % 12 == 1 or k % 12 == 5): sieve[k] ^= True
            k = 3 * x * x + y * y
            if k <= n and k % 12 == 7: sieve[k] ^= True
            k = 3 * x * x - y * y
            if x > y and k <= n and k % 12 == 11: sieve[k] ^= True
            y += 1
        x += 1
    r = 5
    while r * r <= n:
        if sieve[r]:
            for i in range(r * r, n + 1, r * r): sieve[i] = False
        r += 1
    for i in range(5, n + 1):
        if sieve[i]: primes.append(i)
    return primes

def fundamental_lemma_estimate(X: float, primes_up_to_z: list[int]) -> float:
    """
    Calculates the expected number of elements remaining in a set of size X
    after sifting out multiples of primes up to z.

    This function implements the core idea of the Fundamental Lemma of Sieve
    Theory, specifically a Legendre-style estimate for the number of unsifted
    elements. The number of elements not divisible by any prime in a set P is
    estimated as: S(A, P, z) ≈ X * Π_{p≤z} (1 - 1/p).

    Args:
        X: The initial size of the set of numbers.
        primes_up_to_z: A list of primes to be sifted out.

    Returns:
        The estimated count of unsifted elements.
    """
    if not primes_up_to_z:
        return X
    # Use log-sum to prevent underflow
    log_prod = np.sum([math.log(1.0 - 1.0 / p) for p in primes_up_to_z])
    return X * math.exp(log_prod)

def brun_twin_prime_bound(x: int, primes: list[int]) -> float:
    """
    Calculates an upper bound for the number of twin primes up to x using a
    formula derived from Brun's sieve.

    Brun's sieve was a historic result that proved the sum of the reciprocals
    of twin primes converges. A common formulation of the upper bound is:
    π₂(x) ≤ C * x / (log x)² * (1 + O(log log x / log x))

    Args:
        x: The upper limit for counting twin primes.
        primes: A list of primes (not directly used in this formulaic version).

    Returns:
        An upper bound on the number of twin primes up to x.
    """
    twin_prime_constant_C2 = 0.66016
    C = 8 * twin_prime_constant_C2
    log_x = math.log(x)
    if log_x < 1: return x
    return C * x / (log_x**2) * (1 + (math.log(log_x) / log_x))**2

def selberg_twin_prime_bound(x: float) -> float:
    """
    Calculates an upper bound for the number of twin primes up to x
    using the result from the Selberg sieve.

    The Selberg sieve provides a sharp upper bound for sifted sets. For twin
    primes, it gives: π₂(x) ≤ (8 * C₂) * x / (log x)², where C₂ is the twin
    prime constant (≈ 0.66016).

    Args:
        x: The upper limit for counting twin primes.

    Returns:
        A sharp upper bound on the number of twin primes up to x.
    """
    if x < 3: return 0.0
    C = 8 * 0.6601618158  # 8 * Twin Prime Constant C₂
    log_x = math.log(x)
    if log_x < 1e-9: return 0.0
    return C * x / (log_x ** 2)

def logarithmic_integral_li(x: float) -> float:
    """
    Calculates the Logarithmic Integral Li(x), the best-known approximation
    for the prime-counting function π(x).

    Li(x) is defined as the integral of 1/log(t) from 2 to x. We use
    numerical integration (quad) for this calculation.

    Args:
        x: The upper limit of the integral.

    Returns:
        The value of Li(x).
    """
    if x < 2:
        return 0
    result, _ = quad(lambda t: 1 / math.log(t), 2, x)
    return result

def prime_race_mod_4(primes: list[int]) -> tuple[list[int], list[int], list[int]]:
    """
    Calculates the "prime number race" for primes modulo 4.

    This tracks the count of primes of the form 4k+1 versus 4k+3.
    Chebyshev's bias notes that primes of the form 4k+3 are often
    more numerous, though the lead switches infinitely often.

    Args:
        primes: A list of all primes to analyze.

    Returns:
        A tuple containing (prime_x_values, counts_4k1, counts_4k3).
    """
    primes_4k1, primes_4k3 = [], []
    for p in primes:
        if p > 2:
            if p % 4 == 1: primes_4k1.append(p)
            else: primes_4k3.append(p)
    x_values = sorted(list(set(primes_4k1 + primes_4k3)))
    counts_4k1, counts_4k3, c1, c3, idx1, idx3 = [], [], 0, 0, 0, 0
    for p in x_values:
        if idx1 < len(primes_4k1) and primes_4k1[idx1] == p: c1 += 1; idx1 += 1
        if idx3 < len(primes_4k3) and primes_4k3[idx3] == p: c3 += 1; idx3 += 1
        counts_4k1.append(c1); counts_4k3.append(c3)
    return x_values, counts_4k1, counts_4k3

def mertens_second_theorem_sum(primes: list[int]) -> tuple[list[int], list[float], list[float]]:
    """
    Calculates the sum of reciprocals of primes and its theoretical growth.

    Mertens' Second Theorem states that Σ(1/p) for p ≤ x grows like log(log(x)).
    The difference between these two values converges to the Meissel-Mertens
    constant M (≈ 0.261497).

    Args:
        primes: A list of all primes to analyze.

    Returns:
        A tuple containing (valid_primes, actual_sum, theoretical_growth).
    """
    valid_primes = [p for p in primes if p > 1]
    reciprocal_sum = np.cumsum([1.0/p for p in valid_primes])
    theoretical_growth = [math.log(math.log(p)) for p in valid_primes]
    return valid_primes, reciprocal_sum.tolist(), theoretical_growth

def gpy_sieve_check(k: int, N: int, primes: list[int]) -> tuple[bool, float, dict]:
    """
    Implements the core logic of the Goldston-Pintz-Yıldırım (GPY) sieve
    to check for bounded gaps between primes.

    The GPY method shows that S₂ exceeds a specific threshold relative to S₁.
    This implementation uses a simplified ratio (S₂/S₁ > 1) to demonstrate the
    sifting mechanism. In textbooks, if S₂ > (2/θ) * log(k) * S₁, then for infinitely
    many n, the set {n+h₁,...,n+hₖ} contains at least two primes. This was a
    major step towards proving bounded gaps between primes.

    Args:
        k: The size of the tuple. We use a known admissible tuple for this k.
        N: The range of integers `n` to sum over (from N to 2N).
        primes: A list of primes up to 2N + h_k for primality testing.

    Returns:
        A tuple containing (is_condition_met, ratio_S2_S1, parameters).
    """
    # For k=4 and k=6, we use known small admissible tuples.
    # For k=50, we use primes > 50 to guarantee admissibility.
    if k == 4:
        H = [0, 2, 6, 8]
    elif k == 6:
        H = [0, 2, 6, 8, 12, 18]
    elif k == 50:
        all_primes = sieve_of_eratosthenes(1000) # Get plenty of primes
        H = [p for p in all_primes if p > 50][:50]
    else:
        raise ValueError(f"No stored admissible tuple for k={k}.")

    theta = 0.5
    R = N**(theta / 2)
    prime_set = set(primes)

    R_int = int(R)
    mu = [0] * (R_int + 1)
    mu[1] = 1
    primes_small = []
    is_prime_small = [True] * (R_int + 1)
    for i in range(2, R_int + 1):
        if is_prime_small[i]:
            primes_small.append(i)
            mu[i] = -1
        for p in primes_small:
            if i * p > R_int: break
            is_prime_small[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    lambdas = {d: mu[d] * (math.log(R / d)**k) for d in range(1, R_int + 1) if mu[d] != 0}

    S1, S2 = 0.0, 0.0
    for n in range(N, 2 * N):
        lambda_sum = 0.0
        for d, weight in lambdas.items():
            prod = 1
            for h in H:
                prod *= (n + h)
                if prod % d == 0: break
            if prod % d == 0:
                lambda_sum += weight

        if lambda_sum == 0: continue

        weight_sq = lambda_sum**2
        S1 += weight_sq
        prime_count = sum(1 for h in H if (n + h) in prime_set)
        S2 += prime_count * weight_sq

    if S1 == 0: return False, 0.0, {}

    ratio = S2 / S1
    required_ratio = 1.0
    params = {"tuple_H": H, "theta": theta, "N": N, "R": R, "required_ratio": required_ratio}
    return ratio > required_ratio, ratio, params

# ---

def sieve_segmented(n: int) -> list[int]:
    """
    Implements a Segmented Sieve of Eratosthenes to find primes up to n.

    This method is memory-efficient for large n. It first finds all primes up to
    sqrt(n) (the 'base primes'). Then, it sieves the range [1, n] in segments
    of a chosen size, marking multiples of the base primes in each segment.

    Complexity:
    - Time: O(n) (in textbooks, O(n log log n))
    - Space: O(sqrt(n))

    Args:
        n: The upper limit for the sieve (inclusive).

    Returns:
        A list of all prime numbers less than or equal to n.
    """
    if n < 2:
        return []

    limit = int(math.sqrt(n))
    # 1. Generate base primes up to sqrt(n)
    base_primes = []
    is_prime_small = [True] * (limit + 1)
    is_prime_small[0] = is_prime_small[1] = False
    for p in range(2, limit + 1):
        if is_prime_small[p]:
            base_primes.append(p)
            for i in range(p * p, limit + 1, p):
                is_prime_small[i] = False

    # 2. Sieve in segments
    primes = list(base_primes)
    segment_size = limit

    for low in range(limit + 1, n + 1, segment_size):
        high = min(low + segment_size - 1, n)
        is_prime_segment = [True] * (high - low + 1)

        for p in base_primes:
            # Find the starting multiple of p in the current segment
            start_idx = math.ceil(low / p) * p
            if start_idx < low:
                start_idx += p # This should not happen with ceil, but as a safeguard.

            for j in range(start_idx, high + 1, p):
                is_prime_segment[j - low] = False

        for i, is_p in enumerate(is_prime_segment):
            if is_p:
                primes.append(low + i)

    return primes


def sieve_with_wheel(n: int, wheel_primes: list[int] | None = None) -> list[int]:
    """
    Implements the Sieve of Eratosthenes optimized with wheel factorization.

    The wheel pre-eliminates multiples of small primes (e.g., 2, 3, 5). The
    sieve then only needs to check numbers that are coprime to the wheel primes,
    following a repeating pattern of gaps between integers coprime to the wheel product.

    Args:
        n: The upper limit for the sieve.
        wheel_primes: A list of small primes to build the wheel from. Defaults to [2, 3, 5].

    Returns:
        A list of all prime numbers less than or equal to n.
    """
    if wheel_primes is None:
        wheel_primes = [2, 3, 5]
    if n < 2:
        return []

    # 1. Generate the wheel
    primes = [p for p in wheel_primes if p <= n]
    if not wheel_primes:
        return sieve_of_eratosthenes(n) # Fallback to a standard sieve

    wheel_size = np.prod(wheel_primes)
    gaps = []
    is_coprime = [True] * wheel_size
    for p in wheel_primes:
        for i in range(p, wheel_size, p):
            is_coprime[i] = False

    last = 1
    for i in range(2, wheel_size):
        if is_coprime[i]:
            gaps.append(i - last)
            last = i
    gaps.append(wheel_size + 1 - last)

    # 2. Perform the sieve using the wheel
    is_prime = [True] * (n + 1)
    for p in primes:
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False

    k, gap_idx = 1, 0
    while k*k <= n:
        k += gaps[gap_idx]
        gap_idx = (gap_idx + 1) % len(gaps)
        if is_prime[k] and k not in wheel_primes:
            primes.append(k)
            for i in range(k * k, n + 1, k):
                is_prime[i] = False

    # Collect final primes
    while k <= n:
        k += gaps[gap_idx]
        gap_idx = (gap_idx + 1) % len(gaps)
        if k <= n and is_prime[k]:
            primes.append(k)

    primes.sort()
    return primes

def benchmark_sieves(limits: list[int], sieves: dict) -> dict:
    """
    Benchmarks the execution time of different sieve algorithms.

    Args:
        limits: A list of upper bounds (n) to test.
        sieves: A dictionary mapping sieve names to sieve functions.

    Returns:
        A dictionary containing the benchmark results, ready for plotting.
    """
    print("\nStarting sieve benchmark...")
    results = {name: [] for name in sieves}

    for n in limits:
        print(f"  Benchmarking for N = {n}...")
        for name, func in sieves.items():
            start_time = time.time()
            func(n) # Run the sieve
            end_time = time.time()
            results[name].append(end_time - start_time)

    print("Benchmark complete.")
    return {"limits": limits, "results": results}


# ---

def mobius_mu(n: int) -> list[int]:
    """
    Computes the Möbius function μ(k) for all k from 0 to n.

    The Möbius function μ(k) is defined as:
    - μ(k) = 1 if k is a square-free positive integer with an even number of prime factors.
    - μ(k) = -1 if k is a square-free positive integer with an odd number of prime factors.
    - μ(k) = 0 if k has a squared prime factor.
    - μ(0) is typically undefined, we set it to 0.

    Args:
        n: The upper limit.

    Returns:
        A list of Möbius values [μ(0), μ(1), ..., μ(n)].
    """
    mu = [0] * (n + 1)
    lp = [0] * (n + 1) # least prime factor
    primes = []
    mu[1] = 1

    for i in range(2, n + 1):
        if lp[i] == 0:
            lp[i] = i
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if p > lp[i] or i * p > n:
                break
            lp[i * p] = p
            if p == lp[i]:
                mu[i * p] = 0
            else:
                mu[i * p] = -mu[i]
    return mu

def riemann_r(x: float, mu_values: list[int], li_func) -> float:
    """
    Calculates Riemann's prime-counting function R(x).

    R(x) is a very accurate approximation of π(x), defined as:
    R(x) = Σ_{n=1 to ∞} [μ(n)/n] * li(x^(1/n))

    Args:
        x: The value at which to evaluate R(x).
        mu_values: A pre-computed list of Möbius function values.
        li_func: A reference to a function that computes the logarithmic integral li(x).

    Returns:
        The value of R(x).
    """
    if x < 2:
        return 0

    total = 0.0
    n = 1
    while True:
        # The term x^(1/n) gets small quickly. Once it's less than 2, li() is 0.
        term_x = x**(1.0 / n)
        if term_x < 2:
            break
        if n >= len(mu_values):
            # If we need a mu value beyond our pre-computed list, stop.
            # The series converges fast, so this is a safe cutoff for large enough mu_values.
            break

        mu_n = mu_values[n]
        if mu_n == 0:
            n += 1
            continue

        total += (mu_n / n) * li_func(term_x)
        n += 1

    return total

# ---

def goldbach_partitions(n: int, prime_set: set) -> int:
    """
    Counts the number of ways an even integer can be written as the sum of two primes.

    Args:
        n: An even integer greater than 2.
        prime_set: A set of primes for efficient lookup.

    Returns:
        The number of unique pairs (p, q) with p <= q such that p + q = n.
    """
    if n <= 2 or n % 2 != 0:
        return 0
    count = 0
    # Iterate through primes p <= n/2
    for p in sorted(list(prime_set)):
        if p > n / 2:
            break
        q = n - p
        if q in prime_set:
            # If p == q, it's one partition (n = p + p)
            # If p < q, we count it. The loop structure prevents double counting (q, p).
            count += 1
    return count

def analyze_goldbach_comet(max_n: int, primes: list[int]) -> tuple[list, list]:
    """
    Generates the data for the Goldbach Comet plot.

    Args:
        max_n: The maximum even number to check.
        primes: A list of primes up to max_n.

    Returns:
        A tuple of (even_numbers, partition_counts).
    """
    print("\nAnalyzing Goldbach's Conjecture...")
    prime_set = set(primes)
    even_numbers = list(range(4, max_n + 1, 2))
    partition_counts = [goldbach_partitions(n, prime_set) for n in even_numbers]
    print("Goldbach analysis complete.")
    return even_numbers, partition_counts

def count_prime_constellations(gaps: dict[str, int], primes: list[int]) -> dict:
    """
    Counts various prime constellations (twin, cousin, sexy, etc.) up to the
    limit of the provided prime list.

    Args:
        gaps: A dictionary mapping constellation names to their integer gaps.
              e.g., {"Twin (p+2)": 2, "Cousin (p+4)": 4}
        primes: A sorted list of primes.

    Returns:
        A dictionary mapping constellation names to their counts.
    """
    prime_set = set(primes)
    counts = {name: 0 for name in gaps}

    for p in primes:
        for name, gap in gaps.items():
            if (p + gap) in prime_set:
                counts[name] += 1
    return counts

def verify_euler_product(n: int, primes: list[int]):
    """
    Empirically verifies the Euler product formula for the Riemann Zeta function at s=2.

    Compares the partial sum Σ 1/k² with the product Π (1 - 1/p²)^-1 for p <= n.
    Both should converge to ζ(2) = π²/6.

    Args:
        n: The upper limit for the sum and product.
        primes: A list of prime numbers up to n.
    """
    print("\n" + "="*80)
    print(f"Verifying Euler Product Formula for ζ(2) up to N = {n}")
    print("="*80)

    # 1. Calculate the partial sum of the zeta function
    partial_sum = sum(1.0 / (k**2) for k in range(1, n + 1))

    # 2. Calculate the Euler product over primes
    euler_prod = 1.0
    for p in primes:
        if p > n:
            break
        euler_prod *= (1.0 - 1.0 / (p**2))**-1

    # 3. Compare with the known value
    zeta_2 = (math.pi**2) / 6

    print(f"  Known value of ζ(2) = π²/6   ≈ {zeta_2:.8f}")
    print(f"  Partial sum Σ 1/k²           ≈ {partial_sum:.8f}")
    print(f"  Euler product Π (1-p⁻²)⁻¹    ≈ {euler_prod:.8f}")
    print("-" * 40)
    print(f"  Difference (Sum - ζ(2))      ≈ {abs(partial_sum - zeta_2):.8f}")
    print(f"  Difference (Product - ζ(2))  ≈ {abs(euler_prod - zeta_2):.8f}")

# ==============================================================================
# SECTION 2: DATA GENERATION FOR PLOTTING
# ==============================================================================

def generate_plot_data(max_n: int):
    """Generates all necessary data for plotting the sieve results."""
    print(f"Generating data up to N = {max_n}...")
    all_primes = sieve_of_eratosthenes(max_n)
    atkin_primes = sieve_of_atkin(max_n)
    prime_set = set(all_primes)
    x_values = np.linspace(100, max_n, 400, dtype=int)

    pi_x_eratosthenes, pi_x_atkin, pi_x_estimated, pi_x_li = [], [], [], []
    pi_2_x_actual, pi_2_x_brun, pi_2_x_selberg = [], [], []

    for x in x_values:
        count_erat = bisect_right(all_primes, x)
        count_atkin = bisect_right(atkin_primes, x)

        primes_up_to_x = all_primes[:count_erat]
        pi_x_eratosthenes.append(count_erat)
        pi_x_atkin.append(count_atkin)
        pi_x_li.append(logarithmic_integral_li(x))

        # z = int(math.sqrt(x)) # Legendre Sieve, overestimates
        z = int(math.log(x)**2) # Selberg Sieve, valid for small z
        primes_for_sieve = [p for p in primes_up_to_x if p <= z]
        pi_x_estimated.append(fundamental_lemma_estimate(x, primes_for_sieve))

        actual_twins = sum(1 for p in primes_up_to_x if p + 2 in prime_set)
        pi_2_x_actual.append(actual_twins)
        pi_2_x_brun.append(brun_twin_prime_bound(x, primes_for_sieve))
        pi_2_x_selberg.append(selberg_twin_prime_bound(x))

    mu_vals = mobius_mu(int(math.log2(max_n)) + 2)
    pi_x_riemann = [riemann_r(val, mu_vals, logarithmic_integral_li) for val in x_values]
    pi_x_error = (np.array(pi_x_eratosthenes) - np.array(pi_x_li)).tolist()

    pi_2_x_actual_np = np.array(pi_2_x_actual)
    pi2_selberg_diff = np.array(pi_2_x_selberg) - pi_2_x_actual_np
    pi2_brun_diff = np.array(pi_2_x_brun) - pi_2_x_actual_np

    prime_gaps = np.diff(all_primes)
    gap_x_values = np.array(all_primes[1:])
    gap_counts = Counter(prime_gaps)

    race_x, race_4k1, race_4k3 = prime_race_mod_4(all_primes)
    mertens_x, mertens_sum, mertens_theory = mertens_second_theorem_sum(all_primes)
    mertens_diff = (np.array(mertens_sum) - np.array(mertens_theory)).tolist()

    print("Data generation complete.")
    return {
        "x": x_values, "pi_eratosthenes": pi_x_eratosthenes, "pi_atkin": pi_x_atkin,
        "pi_estimated": pi_x_estimated, "pi_li": pi_x_li,
        "pi_riemann": pi_x_riemann,
        "pi_error": pi_x_error,
        "pi2_actual": pi_2_x_actual,
        "pi2_brun": pi_2_x_brun, "pi2_selberg": pi_2_x_selberg,
        "pi2_selberg_diff": pi2_selberg_diff.tolist(), "pi2_brun_diff": pi2_brun_diff.tolist(),
        "prime_gaps": prime_gaps.tolist(), "gap_x_values": gap_x_values.tolist(),
        "gap_counts": gap_counts,
        "race_x": race_x, "race_4k1": race_4k1, "race_4k3": race_4k3,
        "mertens_x": mertens_x, "mertens_sum": mertens_sum, "mertens_theory": mertens_theory,
        "mertens_diff": mertens_diff,
        "all_primes": all_primes
    }

# ==============================================================================
# SECTION 3: PLOTTING WITH PLOTLY
# ==============================================================================

def create_plots(data: dict):
    """Creates and displays enhanced interactive plots using Plotly."""
    print("Creating interactive plots...")
    fig = make_subplots(
        rows=10, cols=1,
        subplot_titles=(
            "<b>Plot 1: Prime Counting Functions π(x)</b>",
            "<b>Plot 2: Prime Counting Error π(x) - Li(x)</b>",
            "<b>Plot 3: Twin Prime Counting Functions π₂(x)</b>",
            "<b>Plot 4: Difference (Bound - Actual) for Twin Primes</b>",
            "<b>Plot 5: Distribution of Gaps Between Primes</b>",
            "<b>Plot 6: Histogram of Prime Gap Frequency</b>",
            "<b>Plot 7: Prime Number Race (mod 4)</b>",
            "<b>Plot 8: Mertens' 2nd Theorem & Meissel-Mertens Constant</b>",
            "<b>Plot 9: Goldbach's Conjecture: Number of Prime Partitions</b>",
            "<b>Plot 10: Performance Benchmark of Prime Sieve Algorithms</b>"
        ),
        vertical_spacing=0.03,
        shared_xaxes=False
    )

    fig.add_trace(go.Scatter(x=data['x'], y=data['pi_eratosthenes'], name="Actual π(x)", line=dict(color='dodgerblue', width=3), hovertemplate='<b>Actual</b><br>x=%{x}<br>π(x)=%{y}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['x'], y=data['pi_li'], name="Li(x) Approx.", line=dict(color='darkviolet', width=2, dash='dash'), hovertemplate='<b>Li(x)</b><br>x=%{x}<br>π(x)≈%{y:.0f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['x'], y=data['pi_riemann'], name="Riemann R(x)", line=dict(color='limegreen', width=2, dash='dot'), hovertemplate='<b>Riemann R(x)</b><br>x=%{x}<br>π(x)≈%{y:.0f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['x'], y=data['pi_estimated'], name="Sieve Estimate", line=dict(color='red', width=2, dash='dot'), hovertemplate='<b>Estimate</b><br>x=%{x}<br>π(x)≈%{y:.0f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['x'], y=data['pi_error'], name="π(x) - Li(x)", line=dict(color='teal'), fill='tozeroy', hovertemplate='<b>Error</b><br>x=%{x}<br>π(x) - Li(x) = %{y:.2f}<extra></extra>'), row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_trace(go.Scatter(x=data['x'], y=data['pi2_actual'], name="Actual π₂(x)", line=dict(color='green', width=3), hovertemplate='<b>Actual</b><br>x=%{x}<br>π₂(x)=%{y}<extra></extra>'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data['x'], y=data['pi2_selberg'], name="Selberg Bound", line=dict(color='purple', width=2, dash='dash'), hovertemplate='<b>Selberg</b><br>x=%{x}<br>Bound≈%{y:.0f}<extra></extra>'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data['x'], y=data['pi2_brun'], name="Brun Bound", line=dict(color='orange', width=2, dash='dashdot'), hovertemplate='<b>Brun</b><br>x=%{x}<br>Bound≈%{y:.0f}<extra></extra>'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data['x'], y=data['pi2_selberg_diff'], name="Selberg - Actual", legendgroup="diff", line=dict(color='purple', width=2), fill='tozeroy', hovertemplate='<b>Selberg Diff</b><br>x=%{x}<br>Diff=%{y:.0f}<extra></extra>'), row=4, col=1)
    fig.add_trace(go.Scatter(x=data['x'], y=data['pi2_brun_diff'], name="Brun - Actual", legendgroup="diff", line=dict(color='orange', width=2), fill='tozeroy', hovertemplate='<b>Brun Diff</b><br>x=%{x}<br>Diff=%{y:.0f}<extra></extra>'), row=4, col=1)
    fig.add_trace(go.Scattergl(x=data['gap_x_values'], y=data['prime_gaps'], name="Prime Gap (pₙ₊₁-pₙ)", mode='markers', marker=dict(color='rgba(220,20,60,0.5)', size=4), legendgroup="gaps", hovertemplate='pₙ=%{x}<br>Gap=%{y}<extra></extra>'), row=5, col=1)
    safe_gap_x = [v for v in data['gap_x_values'] if v > 1]
    log_p_values = np.log(safe_gap_x)
    fig.add_trace(go.Scatter(x=safe_gap_x, y=log_p_values, name="Avg. Gap Trend (log pₙ)", line=dict(color='darkcyan', width=3, dash='dash'), legendgroup="gaps", hovertemplate='pₙ≈%{x}<br>Avg. Gap≈%{y:.2f}<extra></extra>'), row=5, col=1)
    gap_counts_sorted = sorted(data['gap_counts'].items())
    gap_sizes = [str(item[0]) for item in gap_counts_sorted]
    gap_frequencies = [item[1] for item in gap_counts_sorted]
    fig.add_trace(go.Bar(x=gap_sizes, y=gap_frequencies, name="Gap Frequency", marker_color='rgb(26, 118, 255)'), row=6, col=1)
    fig.add_trace(go.Scatter(x=data['race_x'], y=data['race_4k3'], name="Count of 4k+3 Primes", line=dict(color='mediumseagreen', width=2), legendgroup="race", hovertemplate='p≤%{x}<br>Count=%{y}<extra></extra>'), row=7, col=1)
    fig.add_trace(go.Scatter(x=data['race_x'], y=data['race_4k1'], name="Count of 4k+1 Primes", line=dict(color='tomato', width=2), legendgroup="race", hovertemplate='p≤%{x}<br>Count=%{y}<extra></extra>'), row=7, col=1)
    fig.add_trace(go.Scatter(x=data['mertens_x'], y=data['mertens_diff'], name="Σ 1/p - log(log p)", line=dict(color='goldenrod', width=2), legendgroup="mertens", hovertemplate='p≤%{x}<br>Difference≈%{y:.4f}<extra></extra>'), row=8, col=1)
    fig.add_hline(y=0.261497, line_width=2, line_dash="dash", line_color="black", annotation_text="Mertens Constant M ≈ 0.261497", annotation_position="bottom right", row=8, col=1)

    fig.add_trace(go.Scatter(
        x=data['goldbach_x'], y=data['goldbach_y'], mode='markers',
        name='Goldbach Partitions', marker=dict(color='orange', size=5, opacity=0.7),
        hovertemplate='<b>Even number n = %{x}</b><br>Partitions = %{y}<extra></extra>'
    ), row=9, col=1)

    bench_limits = data['benchmark_data']['limits']
    for name, timings in data['benchmark_data']['results'].items():
        fig.add_trace(go.Scatter(
            x=bench_limits, y=timings, name=name, mode='lines+markers', legendgroup="benchmark",
            hovertemplate=f'<b>{name}</b><br>N = %{{x}}<br>Time = %{{y:.4f}}s<extra></extra>'
        ), row=10, col=1)

    fig.update_layout(
        title_text="<b>A Comprehensive Analysis of the Primes</b>", title_x=0.5, title_y=0.99,
        template='plotly_white',
        height=3000,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        font=dict(family="Arial, sans-serif", size=12),
        bargap=0.01
    )

    fig.update_yaxes(title_text="Count of Primes", row=1, col=1)
    fig.update_yaxes(title_text="π(x) - Li(x)", row=2, col=1)
    fig.update_yaxes(title_text="Count of Twin Primes", row=3, col=1)
    fig.update_yaxes(title_text="Bound - Actual Count", row=4, col=1)
    fig.update_yaxes(title_text="Size of Gap", type="log", autorange=True, row=5, col=1)
    fig.update_yaxes(title_text="Frequency", type="log", autorange=True, row=6, col=1)
    fig.update_yaxes(title_text="Cumulative Count", row=7, col=1)
    fig.update_yaxes(title_text="Σ 1/p - log(log p)", row=8, col=1)
    fig.update_yaxes(title_text="Partition Count", row=9, col=1)
    fig.update_yaxes(title_text="Time (s)", type="log", row=10, col=1)

    fig.update_xaxes(title_text="Value of x (or Prime p)", row=8, col=1)

    fig.update_xaxes(title_text="Gap Size", row=6, col=1)
    fig.update_xaxes(title_text="Even Number (n)", row=9, col=1)
    fig.update_xaxes(title_text="Upper Limit (N)", row=10, col=1)

    # Save to a file
    file_path = "prime_analyses_plots.html"
    fig.write_html(file_path)
    full_path = os.path.abspath(file_path)
    print(f"Plot saved to: {full_path}")
    # Open in default browser
    webbrowser.open(f"file://{full_path}")

    # Alternatively:
    # try:
    #     fig.show()
    # except Exception as e:
    #     print(f"Error displaying plot: {e}")

# ==============================================================================
# SECTION 4: CONSTANT CALCULATION
# ==============================================================================

def calculate_and_print_constants(primes: list[int]):
    """Calculates and prints key number theory constants based on the generated primes."""
    print("\n" + "="*80)
    print("Calculating Key Number Theory Constants")
    print("="*80)

    # 1. Twin Prime Constant (C₂) = Π_{p>2} (1 - 1/(p-1)²)
    c2 = 1.0
    for p in primes:
        if p > 2:
            c2 *= (1.0 - 1.0 / ((p - 1.0)**2))
    print(f"Twin Prime Constant (C₂) calculated from primes up to {primes[-1]}:")
    print(f"  Calculated Value ≈ {c2:.6f}")
    print("  Accepted Value   ≈ 0.660162")

    # 2. Meissel-Mertens Constant (M) = γ + Σ [log(1 - 1/p) + 1/p]
    euler_gamma = 0.5772156649
    mertens_sum_part = 0
    for p in primes:
        mertens_sum_part += (math.log(1 - 1.0/p) + 1.0/p)
    m = euler_gamma + mertens_sum_part
    print(f"\nMeissel-Mertens Constant (M) calculated from primes up to {primes[-1]}:")
    print(f"  Calculated Value ≈ {m:.6f}")
    print("  Accepted Value   ≈ 0.261497")

# ==============================================================================
# SECTION 5: MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    MAX_N_PLOT = 50000

    plot_data = generate_plot_data(MAX_N_PLOT)

    benchmark_limits = [10000, 50000, 100000, 200000, 500000, 1000000]
    sieve_functions = {
        "Eratosthenes": sieve_of_eratosthenes, "Atkin": sieve_of_atkin,
        "Segmented": sieve_segmented, "Wheel (2,3,5)": sieve_with_wheel
    }
    plot_data['benchmark_data'] = benchmark_sieves(benchmark_limits, sieve_functions)

    MAX_N_GOLDBACH = MAX_N_PLOT
    all_primes_goldbach = sieve_of_eratosthenes(MAX_N_GOLDBACH)
    comet_x, comet_y = analyze_goldbach_comet(MAX_N_GOLDBACH, all_primes_goldbach)
    plot_data['goldbach_x'] = comet_x
    plot_data['goldbach_y'] = comet_y

    create_plots(plot_data)

    sieve_functions = {
        "Eratosthenes": sieve_of_eratosthenes,
        "Atkin": sieve_of_atkin,
        "Segmented Eratosthenes": sieve_segmented,
        "Wheel (2,3,5)": sieve_with_wheel
    }

    # --- Analyze Prime Constellations ---
    print("\n" + "="*80)
    print(f"Analyzing Prime Constellations up to N = {MAX_N_PLOT}")
    print("="*80)
    constellation_gaps = {
        "Twin Primes (p, p+2)": 2,
        "Cousin Primes (p, p+4)": 4,
        "Sexy Primes (p, p+6)": 6
    }
    constellation_counts = count_prime_constellations(constellation_gaps, plot_data['all_primes'])
    for name, count in constellation_counts.items():
        print(f"  {name}: {count}")

    # --- Verify Euler Product Formula ---
    verify_euler_product(MAX_N_PLOT, plot_data['all_primes'])
    print("\n" + "="*80)
    print("Analysis complete.")
    print("="*80)

    # --- Calculate and Print Constants ---
    calculate_and_print_constants(plot_data['all_primes'])

    # --- Run the GPY Sieve Check ---
    print("\n" + "="*80)
    print("Running GPY Sieve Check for Bounded Gaps Between Primes")
    print("This demonstrates the method, not a counting function.")
    print("="*80)
    try:
        N_GPY, K_GPY = 15000, 50

        if K_GPY == 4: H_temp = [0, 2, 6, 8]
        elif K_GPY == 6: H_temp = [0, 2, 6, 8, 12, 18]
        elif K_GPY == 50:
            all_p = sieve_of_eratosthenes(1000)
            H_temp = [p for p in all_p if p > 50][:50]
        else:
            raise ValueError(f"Unsupported K_GPY value: {K_GPY}")

        primes_for_gpy = sieve_of_eratosthenes(2 * N_GPY + max(H_temp))
        is_proven, ratio, params = gpy_sieve_check(K_GPY, N_GPY, primes_for_gpy)
        print(f"Calculated Ratio S₂/S₁: {ratio:.4f} | Required: > {params['required_ratio']:.4f}")
        if is_proven:
            print("SUCCESS: The tuple contains at least two primes.")
        else:
            print("FAILURE: Ratio too low for this N.")
    except Exception as e:
        print(f"An error occurred during the GPY check: {e}")

print("\nAnalysis complete. Plot HTML file created; Browser opened.")
