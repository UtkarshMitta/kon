# Problem 3: Largest prime factor
# Find the largest prime factor of 600851475143


def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    return all(n % i != 0 for i in range(3, int(n**0.5) + 1, 2))


def largest_prime_factor(n):
    factor = 2
    while n > 1:
        if n % factor == 0 and is_prime(factor):
            largest_factor = factor
        n = n // factor
        if n % factor == 0:
            continue
        factor += 1
    return largest_factor


# Test cases
def test_problem_3():
    assert largest_prime_factor(600851475143) == 6857
    print("Problem 3 tests passed!")


if __name__ == "__main__":
    number = 600851475143
    print(f"Problem 3 Solution: {largest_prime_factor(number)}")
    test_problem_3()
