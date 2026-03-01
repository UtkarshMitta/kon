# Problem 1: Multiples of 3 and 5
# Find the sum of all multiples of 3 or 5 below 1000


def solve_problem_1():
    return sum(num for num in range(1000) if num % 3 == 0 or num % 5 == 0)


# Test cases
def test_problem_1():
    assert solve_problem_1() == 233168


if __name__ == "__main__":
    test_problem_1()
