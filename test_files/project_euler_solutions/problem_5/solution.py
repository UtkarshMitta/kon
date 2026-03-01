def smallest_multiple():
    """Problem 5: Smallest multiple of 2520"""

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
    print(smallest_multiple())
