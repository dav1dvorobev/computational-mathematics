import numpy as np
import plu_decomposition 

def qr(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R

def solve(A, b):
    Q, R = qr(A)
    y = plu_decomposition.solve(Q, b)
    x = plu_decomposition.solve(R, y)
    return x

A = np.array([[1, 4, 7], 
              [2, 5, 8], 
              [3, 6, 10]], dtype=float)
b = np.array([3, 6, 9])
Q, R = qr(A.copy())
print(Q)
print(R)
print('----')
print(np.dot(Q, Q.T))
print('----')

print(A)
print(np.dot(Q, R))
print(solve(A, b))
print(np.linalg.solve(A, b))