# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 01:24:13 2014

@author: pychuang
"""
import numpy as np


def LInfty(V):
    return np.amax(np.abs(V))


def L2norm(V):
    return np.sqrt(np.sum(V**2))


def Init1DLaplaceAb(N):
    '''
    Initialize matrix A and vector b
    for the 1D Laplace equation in centeral difference form.
    Dirchlet boundary conditions are included
    '''
    A = np.zeros((N, N))
    b = np.zeros(N)
    A[0, 0] = 1.0
    for i in range(1, N-1):
        A[i, i-1:i+2] = [1.0, -2.0, 1.0]
    A[N-1, N-1] = 1.0
    return A, b


def InitValue(N, k):
    '''
    Return initial guess of v => v0 = sin(i*k*pi/N), i=0~N
    '''
    return np.array([np.sin(float(i * k) * np.pi / float(N-1))
                    for i in range(N)])
