import numpy as np
import sys
import time

if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} <l>")
    sys.exit(1)

l = int(sys.argv[1])
flops = 0


def split(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = len(M) // 2
    return M[:s, :s], M[:s, s:], M[s:, :s], M[s:, s:]


def stack(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.vstack((
        np.hstack((a, b)),
        np.hstack((c, d))
    ))


def trivial(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ar, ac = A.shape
    br, bc = B.shape

    if ac != br:
        raise ValueError(f"Invalid matrix shapes: A({ar}, {ac}), B({br}, {bc})")

    C = np.zeros((ar, bc))

    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                C[i][j] += A[i][k] * B[k][j]

    global flops
    flops += ar ** 2 + ar ** 2 * 2  # additions + multiplications

    return C


def strassen(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if len(A) <= 2:
        return trivial(A, B)

    a11, a12, a21, a22 = split(A)
    b11, b12, b21, b22 = split(B)

    p1 = multiply(a11 + a22, b11 + b22)
    p2 = multiply(a21 + a22, b11)
    p3 = multiply(a11, b12 - b22)
    p4 = multiply(a22, b21 - b11)
    p5 = multiply(a11 + a12, b22)
    p6 = multiply(a21 - a11, b11 + b12)
    p7 = multiply(a12 - a22, b21 + b22)

    C11 = p1 + p4 - p5 + p7
    C12 = p3 + p5
    C21 = p2 + p4
    C22 = p1 - p2 + p3 + p6

    global flops
    flops += (len(a11) ** 2) * 18

    return stack(C11, C12, C21, C22)


def binet(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ar, _ = A.shape

    if ar < 3:
        return trivial(A, B)

    a = split(A)
    b = split(B)

    C0 = multiply(a[0], b[0]) + multiply(a[1], b[2])
    C1 = multiply(a[0], b[1]) + multiply(a[1], b[3])
    C2 = multiply(a[2], b[0]) + multiply(a[3], b[2])
    C3 = multiply(a[2], b[1]) + multiply(a[3], b[3])

    global flops
    flops = flops + (len(a[0]) ** 2) * 4

    return stack(C0, C1, C2, C3)


def multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ar, ac = A.shape
    br, bc = B.shape

    if ac != br:
        raise ValueError(f"Invalid matrix shapes: A({ar}, {ac}), B({br}, {bc})")

    if ar <= 2 ** l:
        return strassen(A, B)

    return binet(A, B)


for size in range(2, 10):
    flops = 0

    n = 2 ** size
    A = np.random.random((n, n))
    B = np.random.random((n, n))

    st = time.time()
    multiply(A, B)
    et = time.time()

    elapsed_time = et - st
    print(elapsed_time, size, l, flops)
