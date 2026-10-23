import math
import random
import sys
import time

# Remove the limit for integer <-> string conversion for extremely large inputs
sys.set_int_max_str_digits(0)

class LargePrimeEngine:
    """
    Verify and Generate large prime numbers.
    Uses the Miller-Rabin probabilistic primality test, with refinements.
    """

    def is_prime(self, n, k=64):
        """
        Miller-Rabin primality test.
        :param n: Number to test
        :param k: Number of iterations. k=20, 40, 64 are widely accepted
                  as providing a negligible error probability (4^-k).
        :return: True if probably prime, False if composite.
        """
        # Handle small cases and even numbers
        if n <= 1: return False
        if n <= 3: return True
        if n % 2 == 0: return False

        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2

        for _ in range(k):
            # a = random.randint(2, n - 2)
            a = random.randrange(2, n - 1) # Faster
            x = pow(a, d, n) # Modular exponentiation: (a^d) % n
            if x == 1 or x == n - 1:
                continue

            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False # Definitely composite
        return True # Probably prime

    # ---

    def get_small_primes(self, limit=10000):
        """
        Generates a list of primes, up a limit, using Sieve of Eratosthenes.
        Used for generating a pre-sieve filter.
        """
        primes = []
        sieve = [True] * (limit + 1)
        for p in range(2, limit + 1):
            if sieve[p]:
                primes.append(p)
                for i in range(p * p, limit + 1, p):
                    sieve[i] = False
        return primes

    def generate_prime(self, start_value, search_limit=1000000, k=64, pre_sieve_limit=10000):
        """
        Generates the next prime number starting from start_value.
        :param start_value: The integer to start searching from.
        :param search_limit: Max odd numbers to check before giving up.
        :param k: Miller-Rabin robustness constant.
        :param pre_sieve_limit: How many small primes to generate for the initial filter.
        """
        # Ensure we start on an odd number
        current = start_value if start_value % 2 != 0 else start_value + 1

        # Generate the pre-sieve list of small primes (easily pruned)
        small_primes = self.get_small_primes(pre_sieve_limit)
        # Remove '2' because we only check odd numbers
        small_primes = [p for p in small_primes if p > 2]

        attempts = 0
        while attempts < search_limit:
            # 1. Pre-Sieve: Very fast modulo checks
            is_composite = False
            for p in small_primes:
                if current == p: break
                if current % p == 0:
                    is_composite = True
                    break

            if not is_composite:
                # 2. Two-Stage Miller-Rabin for extreme performance:
                # First, do a very quick check (k=2) to omit 99.9% of composites
                if self.is_prime(current, k=2):
                    # Then, do the heavy-duty check for absolute certainty
                    if self.is_prime(current, k=k):
                        return current

            current += 2
            attempts += 1

        raise TimeoutError("Could not find a prime within the search limit.")

    # ---

    def primorial__get_small_primes_product(self, limit=10000):
        """
        Generates all primes up to limit and returns their PRODUCT.
        This allows us to use GCD for near-instant pre-screening.
        """
        sieve = [True] * (limit + 1)
        product = 1
        for p in range(2, limit + 1):
            if sieve[p]:
                product *= p
                for i in range(p * p, limit + 1, p):
                    sieve[i] = False
        return product

    def primorial__generate_prime(self, start_value, search_limit=1000000, k=64, pre_sieve_limit=10000):
        """
        Generates the next prime number starting from start_value, using the GCD trick.
        """
        current = start_value if start_value % 2 != 0 else start_value + 1

        # Optimization: Create one giant product of all small primes
        small_primes_product = self.primorial__get_small_primes_product(pre_sieve_limit)

        attempts = 0
        while attempts < search_limit:
            # 1. The GCD Trick:
            # Instead of a for-loop, one GCD check omits ~90% of candidates instantly.
            if math.gcd(current, small_primes_product) == 1:
                # 2. Two-Stage Miller-Rabin
                if self.is_prime(current, k=2):
                    if self.is_prime(current, k=k):
                        return current

            current += 2
            attempts += 1

        raise TimeoutError("Could not find a prime within the search limit.")

    # ---

    @staticmethod
    def read_number_from_file(file_path):
        try:
            with open(file_path) as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError) as e:
            print(f"File Error: {e}")
            raise

    @staticmethod
    def write_number_to_file(file_path, number):
        with open(file_path, 'w') as f:
            f.write(str(number))


def run(engine, mode, input_file, output_file=None, start_file=None, k=64, pre_sieve_limit=10000):
    """
    Run verify and generate, track runtimes, and handle file I/O.
    """
    print(f"--- Starting {mode} ---")
    start_time = time.perf_counter()

    try:
        if mode == "VERIFY":
            num = engine.read_number_from_file(input_file)
            print(f"Verifying number with {len(str(num))} digits (k={k})...")
            result = engine.is_prime(num, k)
            print(f"Result: {'PRIME' if result else 'COMPOSITE'}")

        elif mode == "GENERATE":
            start_value = engine.read_number_from_file(start_file)
            print(f"Searching for prime starting from {len(str(start_value))} digits (pre_sieve_limit={pre_sieve_limit}, k={k})...")

            # Basic:
            # prime = engine.generate_prime(start_value, k=k, pre_sieve_limit=pre_sieve_limit)

            # Performant:
            prime = engine.primorial__generate_prime(start_value, k=k, pre_sieve_limit=pre_sieve_limit)

            if output_file:
                engine.write_number_to_file(output_file, prime)
            print(f"Prime found! Digits: {len(str(prime))}")
            print(f"Saved to: {output_file}")

    except Exception as e:
        print(f"Runtime Error: {e}")

    end_time = time.perf_counter()
    print(f"Runtime: {end_time - start_time:.4f} seconds\n")

# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    engine = LargePrimeEngine()

    # 1. Create a large_prime.txt file, for verification
    # Example: a known Mersenne Prime: 2^521 - 1. This is a prime with 157 digits.
    large_prime_value = (2**521) - 1
    with open("large_prime.txt", "w") as f:
        f.write(str(large_prime_value))

    # Verify whether prime:
    run(engine, "VERIFY", "large_prime.txt")

    # ---

    # 2. Create a start_value.txt file, for generation
    # Example: a random 200-digit number
    random_start = random.randint(10**199, 10**200)
    with open("start_value.txt", "w") as f:
        f.write(str(random_start))

    # Generate / Find next prime:
    run(engine, "GENERATE", None, "found_prime.txt", "start_value.txt", k=64, pre_sieve_limit=10000)
