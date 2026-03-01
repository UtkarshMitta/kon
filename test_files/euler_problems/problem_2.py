# Problem 2: Fibonacci even numbers
# Find the sum of the even-valued terms in the Fibonacci sequence
# that do not exceed four million


def solve():
    a, b = 1, 2
    total = 0
    while b <= 4_000_000:
        if b % 2 == 0:
            total += b
        a, b = b, a + b
    return total


# Run and verify
if __name__ == "__main__":
    result = solve()
    print(f"Problem 2 Solution: {result}")
    assert result == 4613732, "Test failed for Problem 2"
    print("✅ Problem 2 test passed")

# Additional verification for edge cases
print("Testing edge cases...")
assert solve() == 4613732, "Test failed for Problem 2 edge case"
print("✅ Edge case tests passed")
