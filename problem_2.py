# Problem 2: Fibonacci even numbers
# Find the sum of the even-valued terms in the Fibonacci sequence
# that do not exceed four million


def fibonacci_even_sum():
    a, b = 1, 2
    total = 0
    while b <= 4_000_000:
        if b % 2 == 0:
            total += b
        a, b = b, a + b
    return total


# Test cases
def test_problem_2():
    assert fibonacci_even_sum() == 4613732
    print("Problem 2 tests passed!")


if __name__ == "__main__":
    print(f"Problem 2 Solution: {fibonacci_even_sum()}")
    test_problem_2()
