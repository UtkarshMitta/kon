# Problem 5: Smallest multiple
# Find the smallest positive number that is divisible by all of the numbers 1 to 20


def smallest_multiple():
    result = 1
    for num in range(2, 21):
        while result % num != 0:
            result += 1
    return result


# Test cases
def test_problem_5():
    assert smallest_multiple() == 2520
    print("Problem 5 tests passed!")


if __name__ == "__main__":
    print(f"Problem 5 Solution: {smallest_multiple()}")
    test_problem_5()
