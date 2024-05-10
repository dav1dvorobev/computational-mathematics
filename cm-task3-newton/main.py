import time
import numpy as np
from plu_decomposition import *

def F(x_input):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x_input
    return np.array([
    np.cos(x2 * x1) - np.exp(-3 * x3) + x4 * x5 ** 2 - x6 - np.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440,
    np.sin(x2 * x1) + x3 * x9 * x7 - np.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994,
    x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904,
    2 * np.cos(-x9 + x4) + x5 / (x3 + x1) - np.sin(x2 ** 2) + np.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757,
    np.sin(x5) + 2 * x8 * (x3 + x1) - np.exp(-x7 * (-x10 + x6)) + 2 * np.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862,
    np.exp(x1 - x4 - x9) + x5 ** 2 / x8 + np.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115,
    x2 ** 3 * x7 - np.sin(x10 / x5 + x8) + (x1 - x6) * np.cos(x4) + x3 - 0.7380430076202798014,
    x5 * (x1 - 2 * x6) ** 2 - 2 * np.sin(-x9 + x3) + 0.15e1 * x4 - np.exp(x2 * x7 + x10) + 3.5668321989693809040,
    7 / x6 + np.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499,
    x10 * x1 + x9 * x2 - x8 * x3 + np.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096])

def J(x_input):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x_input
    return np.array([[-x2 * np.sin(x2 * x1), -x1 * np.sin(x2 * x1), 3 * np.exp(-3 * x3), x5 ** 2, 2 * x4 * x5,
                -1, 0, -2 * np.cosh(2 * x8) * x9, -np.sinh(2 * x8), 2],
               [x2 * np.cos(x2 * x1), x1 * np.cos(x2 * x1), x9 * x7, 0, 6 * x5,
                -np.exp(-x10 + x6) - x8 - 1, x3 * x9, -x6, x3 * x7, np.exp(-x10 + x6)],
               [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
               [-x5 / (x3 + x1) ** 2, -2 * x2 * np.cos(x2 ** 2), -x5 / (x3 + x1) ** 2, -2 * np.sin(-x9 + x4),
                1.0 / (x3 + x1), 0, -2 * np.cos(x7 * x10) * x10 * np.sin(x7 * x10), -1,
                2 * np.sin(-x9 + x4), -2 * np.cos(x7 * x10) * x7 * np.sin(x7 * x10)],
               [2 * x8, -2 * np.sin(x2), 2 * x8, 1.0 / (-x9 + x4) ** 2, np.cos(x5),
                x7 * np.exp(-x7 * (-x10 + x6)), -(x10 - x6) * np.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1,
                -1.0 / (-x9 + x4) ** 2, -x7 * np.exp(-x7 * (-x10 + x6))],
               [np.exp(x1 - x4 - x9), -1.5 * x10 * np.sin(3 * x10 * x2), -x6,-np.exp(x1 - x4 - x9),
                2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -np.exp(x1 - x4 - x9), -1.5 * x2 * np.sin(3 * x10 * x2)],
               [np.cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * np.sin(x4), x10 / x5 ** 2 * np.cos(x10 / x5 + x8),
                -np.cos(x4), x2 ** 3, -np.cos(x10 / x5 + x8), 0, -1.0 / x5 * np.cos(x10 / x5 + x8)],
               [2 * x5 * (x1 - 2 * x6), -x7 * np.exp(x2 * x7 + x10), -2 * np.cos(-x9 + x3), 1.5,
               (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * np.exp(x2 * x7 + x10), 0, 2 * np.cos(-x9 + x3),
                -np.exp(x2 * x7 + x10)],
               [-3, -2 * x8 * x10 * x7, 0, np.exp(x5 + x4), np.exp(x5 + x4),
                -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
               [x10, x9, -x8, np.cos(x4 + x5 + x6) * x7, np.cos(x4 + x5 + x6) * x7,
                np.cos(x4 + x5 + x6) * x7, np.sin(x4 + x5 + x6), -x3, x2, x1]])

def basic_newton(x0, tol=1e-6, max_iter=1000): 
    x = x0.copy() 
    operations = 0
    iterations = 1
    start_time = time.time() 
    while iterations < max_iter: 
        dx = solve(J(x), -F(x)) 
        x += dx 
        iterations += 1
        operations += 2*(len(x0)**3)//3
        if np.linalg.norm(dx) < tol: 
            break 
    end_time = time.time() 
    elapsed_time = end_time - start_time 
    print(f'Iterations: {iterations}') 
    print(f'Operations: {operations}') 
    print(f'Time: {elapsed_time}') 
 
def modified_newton(x0, tol=1e-6, max_iter=1000): 
    x = x0.copy() 
    operations = 0
    iterations = 1
    start_time = time.time() 
    Jx = J(x) 
    while iterations < max_iter: 
        dx = solve(Jx, -F(x)) 
        x += dx 
        iterations += 1
        operations += 2*(len(x0)**3)//3
        if np.linalg.norm(dx) < tol: 
            break 
    end_time = time.time() 
    elapsed_time = end_time - start_time 
    print(f'Iterations: {iterations}') 
    print(f'Operations: {operations}') 
    print(f'Time: {elapsed_time}')

def newton(x0, m=1, k=float('inf'), tol=1e-6, max_iter=1000):
    x = x0.copy()
    operations = 0
    iterations = 1
    start_time = time.time()
    Jx = J(x)
    while iterations < max_iter:
        dx = solve(Jx, -F(x))
        x += dx
        operations += 2*(len(x0)**3)//3
        if np.linalg.norm(dx) < tol:
            break
        if iterations < k:
            if iterations % m == 0:
                Jx = J(x)
        iterations += 1
    elapsed_time = time.time() - start_time
    print(f'Iterations: {iterations}')
    print(f'Operations: {operations}')
    print(f'Time: {elapsed_time}')
    print(F(x))
    print(x)

x_start = np.array([0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5])

newton(x_start, m=1, k=float('inf'))
newton(x_start, m=float('inf'), k=1)

# x_start[4] = -0.2