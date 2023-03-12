import numpy as np

global operations_count


def brute_force(A, B):
    #   3 3 3 3 = 12
    # 2 * len(A) ** 3
    # len(A) ** 2 + (2 * len(A)) ** 2
    # n dodawan mnozen
    # 1 0 1 = 1
    # 2 4 8 = 12
    # 3 9 18 = 27
    # len(M) ** 2 +
    flops = 0
    if len(A) == 1:
        flops = 1
    else:
        flops = 12
    return A * B, flops


def split(M):
    n = len(M)
    return M[:n // 2, :n // 2], M[:n // 2, n // 2:], M[n // 2:, :n // 2], M[n // 2:, n // 2:]


def strassen(A, B, flops):
    if len(A) <= 2:
        return brute_force(A, B)
    a, b, c, d = split(A)
    e, f, g, h = split(B)
    p1, flops1 = strassen(a + d, e + h, len(a) ** 2 + len(e) ** 2)  # dodawanie macierzy flopsy = flops += len(A) ** 2
    p2, flops2 = strassen(d, g - e, len(g) ** 2)
    p3, flops3 = strassen(a + b, h, len(a) ** 2)
    p4, flops4 = strassen(b - d, g + h, len(b) ** 2 + len(g) ** 2)
    p5, flops5 = strassen(a, f - h, len(f) ** 2)
    p6, flops6 = strassen(c + d, e, len(c) ** 2)
    p7, flops7 = strassen(a - c, e + f, len(a) ** 2 + len(e) ** 2)
    flops += flops1 + flops2 + flops3 + flops4 + flops5 + flops6 + flops7
    C11 = p1 + p2 - p3 + p4
    C12 = p5 + p3
    C21 = p6 + p2
    C22 = p5 + p1 - p6 + p7
    return np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22)))), flops


_, flops = strassen(np.ones((3,3)), np.ones((3,3)), 0)
print(flops)

