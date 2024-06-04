import numpy as np

def jacobi(A, b, x0, tol=1e-6, max_iterations=1000):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D_inv = np.linalg.inv(D)
    x = x0
    for k in range(max_iterations):
        x_new = np.dot(D_inv, (b - np.dot((L + U), x)))
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, k
        x = x_new
    return x, max_iterations

def gauss_seidel(A, b, x0, tol=1e-6, max_iterations=1000):
    n = len(A)
    x = x0
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, k+1
        x = x_new
    return x, max_iterations

def generate_diagonally_dominant_matrix(n):
    A = np.random.rand(n, n)
    for i in range(n):
        A[i, i] += sum(np.abs(A[i]))
    return A

def generate_positive_definite_matrix(n):
    A = np.random.rand(n, n)
    return A.T @ A

n = 3
x0 = np.zeros(n)

# СЛАУ с диагональным преобладанием
A = generate_diagonally_dominant_matrix(n)
b = np.random.rand(n)

x_jacobi, iter_jacobi = jacobi(A, b, x0)
x_gauss_seidel, iter_gauss_seidel = gauss_seidel(A, b, x0)

print(f"Jacobi (diagonally dominant): {iter_jacobi} iterations")
print(f'x = {x_jacobi}')
print(f"Gauss-Seidel (diagonally dominant): {iter_gauss_seidel} iterations")
print(f'x = {x_gauss_seidel}')

print('-----------------------------------------------------------')

# СЛАУ с положительно определённой матрицей
A = generate_positive_definite_matrix(n)
b = np.random.rand(n)
x_jacobi, iter_jacobi = jacobi(A, b, x0)
x_gauss_seidel, iter_gauss_seidel = gauss_seidel(A, b, x0)

print(f"Jacobi (positive definite): {iter_jacobi} iterations")
print(f'x = {x_jacobi}')
print(f"Gauss-Seidel (positive definite): {iter_gauss_seidel} iterations")
print(f'x = {x_gauss_seidel}')

'''
Априорная оценка числа итераций
чето-там со спектральным радиусом =\ 
Сравнение апостериорной оценки
Апостериорная оценка числа итераций основана на фактическом числе итераций, необходимых для достижения заданной точности.
В данном примере можно заметить, что матрицы с диагональным преобладанием обычно быстрее сходятся, 
особенно в методе Зейделя, в то время как положительно определённые матрицы могут потребовать больше итераций.
Таким образом, проведя описанные выше шаги, можно выполнить сравнительный анализ эффективности методов Якоби и Зейделя для различных типов матриц.
'''