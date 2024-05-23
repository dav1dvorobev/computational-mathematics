import numpy as np

def f(x) : return 4 * np.cos(0.5 * x) * np.exp(-5 * x / 4) + 2 * np.sin(4.5 * x) * np.exp(x / 8) + 2

a = 1.3
b = 2.2
alpha = 0
beta = 5/6
answer = 10.83954510946909397740794566485262705081

def u_0(x): return -6 * (2.2 - x) ** (1/6)
def u_1(x): return -6 * (2.2 - x) ** (1/6) * (5*x + 66)/35
def u_2(x): return -6 * (2.2 - x) ** (1/6) * (175 * x**2 + 660*x + 8712)/2275
def u_3(x): return -6 * (2.2 - x) ** (1/6) * (11375 * x**3 + 34650 * x**2 + 130680*x + 1724976)/216125
def u_4(x): return -6 * (2.2 - x) ** (1/6) * (1080625 * x**4 + 3003000 * x**3 + 9147600 * x**2 + 34499520*x + 455393664)/27015625
def u_5(x): return -6 * (2.2 - x) ** (1/6) * (27015625 * x**5 + 71321250 * x**4 + 198198000 * x**3 + 603741600 * x**2 + 2276968320*x + 30055981824)/837484375

def ikf(a, b):
    moments = [u_0(b) - u_0(a), u_1(b) - u_1(a), u_2(b) - u_2(a)]
    node = np.array([a, (a + b) / 2, b])
    xs = np.array([node ** 0, node, node ** 2], dtype=float)
    A = np.linalg.solve(xs, moments)
    return sum(A[i] * f(node[i]) for i in range(3))

def gkf(a, b):
    moments = [u_0(b) - u_0(a), u_1(b) - u_1(a), u_2(b) - u_2(a),
               u_3(b) - u_3(a), u_4(b) - u_4(a), u_5(b) - u_5(a)]
    A = np.array(
        [[moments[0], moments[1], moments[2]],
         [moments[1], moments[2], moments[3]],
         [moments[2], moments[3], moments[4]]], dtype=float)
    b = np.array(moments[3:])
    coeffs = np.linalg.solve(A, -b)
    a = np.concatenate(([1], coeffs[::-1]))
    roots = np.roots(a)
    A = np.linalg.solve(np.array([roots**0, roots, roots**2]), moments[:3])
    return sum(A[i] * f(roots[i]) for i in range(3))

def skf(a, b, formulatype, k=1, L=2, m=2, tol=1e-6, log_output=False):
    integrals = []
    h_i = []
    while True:
        solution = 0
        h = (b - a) / k
        for i in range(k):
            x0 = a + i * h
            x1 = x0 + h
            solution += formulatype(x0, x1)
        integrals.append(solution)
        h_i.append(h)
        if len(integrals) > 1:
            
            # Aitken acceleration
            if len(integrals) > 2:
                m = -np.log(abs((integrals[-1] - integrals[-2]) / (integrals[-2] - integrals[-3]))) / np.log(L)
            log_output and print(m)

            # Richardson method
            r = len(h_i)
            coefficient_matrix = np.zeros((r, r))
            for i in range(r):
                coefficient_matrix[i, 0] = 1
                for j in range(1, r):
                    coefficient_matrix[i, j] = -(h_i[i] ** (m + j - 1))
            S_h = np.array(integrals, dtype=float)
            C_m = np.linalg.solve(coefficient_matrix, S_h)
            error = abs(C_m[0] - S_h[-1])
            print(error, abs(S_h[-1] - answer))
            if error < tol:
                break
        k *= L
    return integrals

def k_opt(integrals):
    L = 2
    m = -np.log((integrals[2] - integrals[1]) / (integrals[1] - integrals[0])) / np.log(L)
    h_1 = (b - a) / 1
    h_opt = 0.95 * h_1 * (1e-6 * (1 - L**(-m)) / abs(integrals[1] - integrals[0])) ** (1/m)
    return int(np.ceil((b - a)/h_opt))

# ikf_integral = ikf(a, b)

# M_n = 240
# integral = 0.06569040766966083
# methodical_error = (M_n / 6) * integral

# exact_error = abs(answer - ikf_integral)
# print(f'ikf integral error - - > {exact_error}')

# gkf_integral = gkf(a, b)
# exact_error = abs(answer - gkf_integral)
# print(f'gkf integral error - - > {exact_error}')

integrals = skf(a, b, formulatype=ikf, log_output=True)
# print(skf(a, b, formulatype=ikf, k=k_opt(integrals), log_output=True))
