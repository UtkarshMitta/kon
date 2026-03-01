# Problem 4: Largest palindrome product
# Find the largest palindrome made from the product of two 3-digit numbers


def is_palindrome(n):
    s = str(n)
    return s == s[::-1]


def solve():
    max_product = 0
    for i in range(999, 99, -1):
        for j in range(i, 99, -1):
            product = i * j
            if product > max_product and is_palindrome(product):
                max_product = product
    return max_product


# Run and verify
if __name__ == "__main__":
    result = solve()
    print(f"Problem 4 Solution: {result}")
    assert result == 906609, "Test failed for Problem 4"
    print("✅ Problem 4 test passed")
