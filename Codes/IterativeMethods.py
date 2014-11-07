# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 12:39:30 2014

@author: pychuang
"""
import numpy as np


# Jacobi Method
def Jacobi(N, A, v0, b, Niter=1):
    '''
    Inputs:
        N: the dimension of A, v0, and b (integer)
        A: coefficient matrix (N by N array)
        v0: initial guess (N by 1 array)
        b: constant matrix (N by 1 array)
        Niter: the number of iterations needed to be performed
    Return:
        The approximated solution v after Niter iterations
        (N by 1 array)
    '''
    v = np.empty(N)
    DA = np.diag(A)
    for ITER in range(Niter):
        v = v0 + (b - np.dot(A, v0)) / DA
        v0 = v.copy()
    return v


# Weighted Jacobi Method
def WJacobi(N, A, v0, b, omg=2./3., Niter=1):
    '''
    Inputs:
        N: the dimension of A, v0, and b (integer)
        A: coefficient matrix (N by N array)
        v0: initial guess (N by 1 array)
        b: constant matrix (N by 1 array)
        omg: relaxation parameter (float)
        Niter: the number of iterations needed to be performed
    Return:
        The approximated solution v after Niter iterations
        (N by 1 array)
    '''
    v1 = np.empty(N)
    DA = np.diag(A)
    for ITER in range(Niter):
        v1 = v0 + omg * (b - np.dot(A, v0)) / DA
        v0 = v1.copy()
    return v0


# Gauss-Seidel Method
def GaussSeidel(N, A, v0, b, Niter=1):
    '''
    Inputs:
        N: the dimension of A, v0, and b (integer)
        A: coefficient matrix (N by N array)
        v0: initial guess (N by 1 array)
        b: constant matrix (N by 1 array)
        Niter: the number of iterations needed to be performed
    Return:
        The approximated solution v after Niter iterations
        (N by 1 array)
    '''
    Ad = np.diag(A)
    for ITER in range(Niter):
        for i in range(N):
            v0[i] = v0[i] + (b[i] - np.dot(A[i, :], v0)) / Ad[i]
    return v0


# Red-Black Gauss-Seidel Method
def RBGaussSeidel(N, A, v0, b, Niter=1):
    '''
    Inputs:
        N: the dimension of A, v0, and b (integer)
        A: coefficient matrix (N by N array)
        v0: initial guess (N by 1 array)
        b: constant matrix (N by 1 array)
        Niter: the number of iterations needed to be performed
    Return:
        The approximated solution v after Niter iterations
        (N by 1 array)
    '''
    Ad = np.diag(A)
    for ITER in range(Niter):
        v0[0::2] = v0[0::2] + \
            (b[0::2] - np.dot(A[0::2, :], v0)) / Ad[0::2]
        v0[1::2] = v0[1::2] + \
            (b[1::2] - np.dot(A[1::2, :], v0)) / Ad[1::2]
    return v0
