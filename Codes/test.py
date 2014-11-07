import numpy as np
import matplotlib.pyplot as plt
from IterativeMethods import *
from Miscellaneous import *

def C2F1D(NC, vC):
    NF = 2 * NC - 1
    vF = np.empty(NF)
    for i in range(NC-1):
        vF[2*i] = vC[i]
        vF[2*i+1] = (vC[i] + vC[i+1]) * 0.5
    vF[-1] = vC[-1]
    return vF

N1 = 4
N2 = 7

x1 = np.linspace(0., 1., N1)
x2 = np.linspace(0., 1., N2)
x_exact = np.linspace(0., 1., 101)

v1 = InitValue(N1, 2)
v2 = C2F1D(N1, v1)
v_exact = InitValue(101, 2)

plt.plot(x1, v1, 'r^--', lw=2, markersize=14,
         label='Values on Coarse Grid')
plt.plot(x2, v2, 'bo-.', lw=2, markersize=8,
         label='Values on Fine Grid')
plt.plot(x_exact, v_exact, 'k-', lw=2,
         label='Exact Solution')
plt.grid()
plt.legend()
plt.show()
