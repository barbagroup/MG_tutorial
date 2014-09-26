# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 02:57:31 2014

@author: pychuang
"""
import numpy as np
import matplotlib.pyplot as plt
from WJacobi import WJacobi
from GaussSeidel import GaussSeidel
from RBGaussSeidel import RB_GaussSeidel
from plotErr import plotError


def initGuess(n, k):
    return np.array([np.sin(float(j * k) * np.pi / float(n))
                     for j in range(n+1)])

w = 2.0 / 3.0
n = 64
k = [1, 3, 6]

A = np.diag(np.ones(n+1)*2, 0) + np.diag(-np.ones(n), 1) + \
    np.diag(-np.ones(n), -1)
v = np.empty_like(n+1)
b = np.zeros(n+1)

eWJ = np.empty((3, 101))
eGS = np.empty((3, 101))
eRB = np.empty((3, 101))

for m in range(3):
    v0 = initGuess(n, k[m])
    eWJ[m, 0] = np.sum(v0**2)**0.5
    eGS[m, 0] = np.sum(v0**2)**0.5
    eRB[m, 0] = np.sum(v0**2)**0.5

    v = v0.copy()
    for i in range(1, 101):
        v = WJacobi(A, v.copy(), b, w, 1)
        eWJ[m, i] = (np.sum(v**2)**0.5) / eWJ[m, 0]

    v = v0.copy()
    for i in range(1, 101):
        v = GaussSeidel(A, v.copy(), b, 1)
        eGS[m, i] = (np.sum(v**2)**0.5) / eGS[m, 0]

    v = v0.copy()
    for i in range(1, 101):
        v = RB_GaussSeidel(A, v.copy(), b, 1)
        eRB[m, i] = (np.sum(v**2)**0.5) / eRB[m, 0]

eWJ[:, 0] = 1.0
eGS[:, 0] = 1.0
eRB[:, 0] = 1.0

L = ['k-.', 'k--', 'k-']
label = ['k = 1', 'k = 3', 'k = 6']
plotError(eWJ, 'Weighted Jacobi', label, L, 0)
plotError(eGS, 'Gauss-Seidel', label, L, 0)
plotError(eRB, 'Red-Black Gauss-Seidel', label, L, 0)
plotError(eWJ, 'Weighted Jacobi_Log', label, L, 1)
plotError(eGS, 'Gauss-Seidel_Log', label, L, 1)
plotError(eRB, 'Red-Black Gauss-Seidel_Log', label, L, 1)


v0 = (initGuess(n, 1) + initGuess(n, 6) + initGuess(n, 32)) / 3.0
eWJ[0, 0] = np.amax(np.abs(v0))
v = v0.copy()
for i in range(1, 101):
    v = WJacobi(A, v.copy(), b, w, 1)
    eWJ[0, i] = np.amax(np.abs(v))
plotError(eWJ[0, :], 'Weighted Jacobi 2',
          ['k = (1 + 6 + 32) / 3'], ['k-'], 0)
plt.show()