import numpy as np

def rank(A):
    M = A.copy()
    # Приводим матрицу к ступенчатой форме
    for i in range(min(M.shape)):
        # Поиск ведущего элемента по столбцу
        pivot_row = i
        for j in range(i+1, M.shape[0]):
            if abs(M[j, i]) > abs(M[pivot_row, i]):
                pivot_row = j
        M[[i, pivot_row]] = M[[pivot_row, i]]
        # Обнуление элементов под диагональю в столбце i
        for j in range(i+1, M.shape[0]):
            if M[i, i] != 0:
                k = M[j, i] / M[i, i]
                M[j, i:] -= k * M[i, i:]
    rank = np.sum(np.abs(M.diagonal()) > 1e-6)
    # Проверка на совместность и нахождение частного решения
    if rank < M.shape[1]:
        print("Система с вырожденной матрицей совместна.")
        print("Любое частное решение:")
        x = np.zeros(M.shape[1])
        for i in range(rank):
            x[i] = M[i, -1] / M[i, i]
        print(x)
    else:
        print("Система с вырожденной матрицей несовместна.")
    return rank

A = np.array([
[1, 1,  1],
[1, -1, 2],
[2, -1, -1]], dtype=float)
print("Ранг матрицы A:", rank(A))