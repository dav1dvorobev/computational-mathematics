import math
import toleranceMath

def f_exact(x: float) -> float: return math.sqrt(math.sinh(2*x + 0.45)) * math.atan(6*x + 1)
def f_approx(x: float) -> float: return toleranceMath.sqrt(toleranceMath.sinh(2*x + 0.45)) * toleranceMath.atan(6*x + 1)

start: int = int(0.01  * 1e3) # remove the type tolerance
end:   int = int(0.06  * 1e3)
step:  int = int(0.005 * 1e3)

print('\tx\t|\t\t f_exact \t\t|\t\t f_approx \t\t| error', '-' * 63, sep = '\n') # head
for x in range(start, end + 1, step):
    x /= 1e3
    print(f'{x} \t| {f_exact(x)} \t| {f_approx(x)} \t| {1e-10 < abs(f_exact(x) - f_approx(x)) < 1e-6}')