# Problem 1: Multiples of 3 and 5
# Find the sum of all multiples of 3 or 5 below 1000


def solve():
    return sum(num for num in range(1000) if num % 3 == 0 or num % 5 == 0)


# Run and verify
if __name__ == "__main__":
    result = solve()
    print(f"Problem 1 Solution: {result}")
    assert result == 233168, "Test failed for Problem 1"
    print("✅ Problem 1 test passed")
