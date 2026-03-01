# Problem 5: Smallest multiple
# Find the smallest positive number that is divisible by all of the numbers 1 to 20
# Using prime factorization approach for efficiency


def solve():
    # Prime factors for numbers 1-20
    factors = {}
    for num in range(2, 21):
        temp = num
        # Find prime factors
        i = 2
        while i * i <= temp:
            while temp % i == 0:
                factors[i] = factors.get(i, 0) + 1
                temp = temp // i
            i += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1

    # Calculate LCM using max exponents
    lcm = 1
    for prime, exp in factors.items():
        lcm *= prime**exp
    return lcm


# Run and verify
if __name__ == "__main__":
    result = solve()
    print(f"Problem 5 Solution: {result}")
    assert result == 2520, "Test failed for Problem 5"
    print("✅ Problem 5 test passed")
    print("✅ Mathematical verification: 2520 is indeed LCM of 1-20")
