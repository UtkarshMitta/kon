# Problem 4: Largest palindrome product
# Find the largest palindrome made from the product of two 3-digit numbers


def is_palindrome(n):
    s = str(n)
    return s == s[::-1]


def solve_problem_4():
    max_product = 0
    for i in range(999, 99, -1):
        for j in range(i, 99, -1):
            product = i * j
            if product > max_product and is_palindrome(product):
                max_product = product
    return max_product


# Test cases
def test_problem_4():
    assert solve_problem_4() == 906609
    print("Problem 4 tests passed!")


if __name__ == "__main__":
    print(f"Problem 4 Solution: {solve_problem_4()}")
    test_problem_4()
