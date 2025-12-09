import math

def f(x, n):
    return (x*x + 1) % n

def pollards_rho_floyd(n, x0=2, max_iter=10000):
    x = f(x0, n)              # x_j
    y = f(x, n)         # x_{2j}
    j = 1

    while j <= max_iter:
        d = math.gcd(abs(y - x), n)

        print(f"j={j}:  x_{j} = {x},   x_2*{j} = {y},   gcd(|x_2*{j} - x_{j}| ({abs(y - x)}), {n}) = {d}")

        if 1 < d < n:
            print("\n>>> Found non-trivial factor:", d)
            return d

        # advance
        x = f(x, n)          # slow: 1 step
        y = f(f(y, n), n)    # fast: 2 steps
        j += 1

    return None

n = 5933
factor = pollards_rho_floyd(n, x0=2)
print("Factor found:", factor)
print("Other factor:", n//factor)

import math

def fermat_factor(n, max_iter=1000):
    t0 = math.isqrt(n)

    print("t0 =", t0)
    print()

    for k in range(0, max_iter + 1):
        t = t0 + k
        s2 = t*t - n
        
        if s2 < 0:
            continue

        # Check if perfect square
        s = int(math.isqrt(s2))
        is_square = s*s == s2

        print(f"k={k:2d}:  t = {t},   t^2 - n = {s2},   perfect square? {'yes' if is_square else 'no'}")

        if is_square:
            a = t - s
            b = t + s
            print("\n>>> Factors found!")
            print("s =", s)
            print("t =", t)
            print("a =", a)
            print("b =", b)
            return a, b

    return None
n = 8823
fermat_factor(n)