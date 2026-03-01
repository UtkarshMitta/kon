def largest_palindrome_product():
    """Problem 4: Largest palindrome product of two 3-digit numbers"""
    max_palindrome = 0
    for i in range(999, 99, -1):
        for j in range(i, 99, -1):
            product = i * j
            if str(product) == str(product)[::-1] and product > max_palindrome:
                max_palindrome = product
    return max_palindrome


if __name__ == "__main__":
    print(largest_palindrome_product())
