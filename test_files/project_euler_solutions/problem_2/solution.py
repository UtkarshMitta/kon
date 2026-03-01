def even_fibonacci_numbers():
    """Problem 2: Even Fibonacci numbers below 4,000,000"""
    a, b = 1, 2
    total = 0
    while b < 4_000_000:
        if b % 2 == 0:
            total += b
        a, b = b, a + b
    return total


if __name__ == "__main__":
    print(even_fibonacci_numbers())
