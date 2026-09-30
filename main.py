import math

import numpy as np

from autodiff import Node, D


e = Node(np.e)
comp = Node(1j)

i = Node(1j)


def sin(x: Node) -> Node:
    return ((i * x).exp() - (-i * x).exp()) / (2 * i)


def taylor(func_arg: tuple[callable, np.ndarray], n: int, a: float):
    result = []
    func = func_arg[0]
    x = np.asanyarray(func_arg[1])

    for k in range(n + 1):
        result.append(D[k](func)(a) / math.factorial(k) * (x - a) ** k)
    return result


def maclaurin(func: tuple[callable, np.ndarray], n: int) -> Node:
    return taylor(func, n, a=0.0)


D[2](lambda x: x * x)(3.0)  # 本来 2
x = Node(np.pi / 4)
n = 10
taylor_series = maclaurin((sin, x), n)

print(f"sin(0) = sum [{', '.join(map(str, taylor_series))}]")
print(f"sin({x}) = sum^{n} {taylor_series}")
print(f"sin({x}) = {sum(taylor_series)}")


# A = Node(np.array([2**3, 2**4, 2**5]))

# print(f"A = {A}")

# print(f"exp(A) = {A.exp()}")
# print(f"sin(A) = {sin(A)}")
# print(f"sin(A)' = {D[1](sin)(A)}")


x = Node(3.0)

y = x * x

print(D[0](lambda x: x * x)(3.0))  # 9
print(D[1](lambda x: x * x)(3.0))  # 6
print(D[2](lambda x: x * x)(3.0))  # 本来 2
print(D[3](lambda x: x * x)(3.0))  # 本来 0
