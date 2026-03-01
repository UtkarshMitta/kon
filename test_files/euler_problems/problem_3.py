# Problem 3: Largest prime factor
# Find the largest prime factor of 600851475143


def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def solve():
    n = 600851475143
    largest_factor = 1
    factor = 2
    while n > 1:
        while n % factor == 0:
            largest_factor = factor
            n = n // factor
        factor += 1
    return largest_factor


# Run and verify
if __name__ == "__main__":
    result = solve()
    print(f"Problem 3 Solution: {result}")
    assert result == 6857, "Test failed for Problem 3"
    print("✅ Problem 3 test passed")
