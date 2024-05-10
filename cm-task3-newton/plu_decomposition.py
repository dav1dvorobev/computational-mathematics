import numpy as np

def plu(A):
    n = A.shape[0]
    P = np.eye(n)
    L = np.eye(n)
    U = A.copy()
    for j in range(n - 1):
        max_index = np.argmax(np.abs(U[j:, j])) + j
        if max_index != j:
            U[[j, max_index]] = U[[max_index, j]]
            P[[j, max_index]] = P[[max_index, j]]
            if j > 0:
                L[[j, max_index], :j] = L[[max_index, j], :j]
        for i in range(j + 1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] -= L[i, j] * U[j, j:]
    return P, L, U

def solve(A, b):
    P, L, U = plu(A)
    n = A.shape[0]
    x = np.zeros(n)
    y = np.zeros(n)
    Pb = np.dot(P, b)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x