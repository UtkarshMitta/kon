def multiples_of_3_and_5():
    """Problem 1: Multiples of 3 and 5 below 1000"""
    return sum(i for i in range(1000) if i % 3 == 0 or i % 5 == 0)


if __name__ == "__main__":
    print(multiples_of_3_and_5())
