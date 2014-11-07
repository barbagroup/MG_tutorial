# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 02:38:36 2014

@author: pychuang
"""
import matplotlib.pyplot as plt
import numpy as np
from WJacobi import WJacobi


def initialize(n, k):
    return np.sin(np.array(range(n+1)) * k * np.pi / n)

case = 'U(x) after 8 iterations'

plt.figure(case, dpi=100)
plt.title(case)

nh = 64
n2h = int(nh / 2)
w = 2.0 / 3.0
ct = 2

v0h = 0.5 * (initialize(nh, 16) + initialize(nh, 40))

plt.plot(np.array(range(nh+1))/(nh+1), v0h, 'r-.', lw=2, label=r'$U_{initial}$')

bh = np.zeros(nh+1)
eh = np.empty(nh+1)
Ah = np.diag(np.ones(nh+1) * 2, 0) + \
    np.diag(- np.ones(nh), 1) + \
    np.diag(- np.ones(nh), -1)

e02h = np.zeros(n2h+1)
r2h = np.zeros(n2h+1)
A2h = np.diag(np.ones(n2h+1) * 2, 0) + \
    np.diag(- np.ones(n2h), 1) + \
    np.diag(- np.ones(n2h), -1)
A2h = A2h / 4.0

vh = WJacobi(Ah, v0h[:], bh, w, ct)

rh = bh - np.dot(Ah, vh)
r2h[0] = rh[0]
r2h[n2h] = rh[nh]
for i in range(1, n2h):
    r2h[i] = 0.25 * (rh[2*i-1] + 2 * rh[2*i] + rh[2*i+1])

e2h = WJacobi(A2h, e02h[:], r2h, w, ct)

eh[nh] = e2h[n2h]
for i in range(n2h):
    eh[2*i] = e2h[i]
    eh[2*i+1] = 0.5 * (e2h[i] + e2h[i+1])

vh = vh + eh
vh = WJacobi(Ah, vh[:], bh, w, ct)

rh = bh - np.dot(Ah, vh)
r2h[0] = rh[0]
r2h[n2h] = rh[nh]
for i in range(1, n2h):
    r2h[i] = 0.25 * (rh[2*i-1] + 2 * rh[2*i] + rh[2*i+1])

e2h = WJacobi(A2h, e02h[:], r2h, w, ct)

eh[nh] = e2h[n2h]
for i in range(n2h):
    eh[2*i] = e2h[i]
    eh[2*i+1] = 0.5 * (e2h[i] + e2h[i+1])

vh = vh + eh

plt.plot(np.array(range(nh+1))/(nh+1), vh, 'k-', lw=3, label=r'$U_{MG}$')


vh = WJacobi(Ah, v0h[:], bh, w, ct*4)
plt.plot(np.array(range(nh+1))/(nh+1), vh, 'b--', lw=3, label=r'$U_{WJ}$')

plt.xlim((0.0, 1.0))
plt.ylim((-1.0, 1.0))
plt.xlabel(r'$x (non-dimensional)$')
plt.ylabel((r'$U (non-dimensional)$'))
plt.yticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.grid()
plt.legend(loc=0)
plt.savefig(case + '.png', dpi=200)
plt.show()