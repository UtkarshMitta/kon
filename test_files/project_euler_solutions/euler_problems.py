def problem_1():
    """Multiples of 3 and 5 below 1000"""
    return sum(i for i in range(1000) if i % 3 == 0 or i % 5 == 0)


def problem_2():
    """Even Fibonacci numbers below 4,000,000"""
    a, b = 1, 2
    total = 0
    while b < 4_000_000:
        if b % 2 == 0:
            total += b
        a, b = b, a + b
    return total


def problem_3():
    """Largest prime factor of 600851475143"""

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    num = 600851475143
    largest_factor = 1
    for i in range(2, int(num**0.5) + 1):
        while num % i == 0:
            largest_factor = i
            num //= i
    if num > 1:
        largest_factor = num
    return largest_factor


def problem_4():
    """Largest palindrome product of two 3-digit numbers"""
    max_palindrome = 0
    for i in range(999, 99, -1):
        for j in range(i, 99, -1):
            product = i * j
            if str(product) == str(product)[::-1] and product > max_palindrome:
                max_palindrome = product
    return max_palindrome


def problem_5():
    """Smallest multiple of 2520"""

    def lcm(a, b):
        return a * b // gcd(a, b)

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    result = 2520
    for i in range(2, 21):
        result = lcm(result, i)
    return result


if __name__ == "__main__":
    print("Problem 1:", problem_1())
    print("Problem 2:", problem_2())
    print("Problem 3:", problem_3())
    print("Problem 4:", problem_4())
    print("Problem 5:", problem_5())
