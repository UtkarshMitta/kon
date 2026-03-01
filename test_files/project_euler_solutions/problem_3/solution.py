def largest_prime_factor():
    """Problem 3: Largest prime factor of 600851475143"""

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


if __name__ == "__main__":
    print(largest_prime_factor())
