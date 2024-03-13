#toleranceMath

import math
from functools import lru_cache

tolerance:     float = 1e-6
sinhTolerance: float = tolerance/2.04
atanTolerance: float = tolerance/2.37
sqrtTolerance: float = tolerance/2.82 

@lru_cache
def cache_factorial(x: int) -> int: return x * cache_factorial(x - 1) if x > 0 else 1

def sinh(x: float) -> float:
    output = 0.0
    for k in range(0, int(1e6) + 1):
        cur = x**(2*k + 1) / cache_factorial(2*k + 1)
        output += cur
        if(abs(cur) <= sinhTolerance): return output

def atan(x: float) -> float:
    output = math.pi/2 
    for k in range(0, int(1e6) + 1):
        cur = (-1)**(k) * x**-(2*k + 1)/(2*k + 1)
        output -= cur
        if(abs(cur) <= atanTolerance): return output

def sqrt(x: float) -> float:
    last = 1.0
    while(True):
         next = 0.5 * (last + x/last)
         if(abs(next - last) <= sqrtTolerance): return next
         last = next