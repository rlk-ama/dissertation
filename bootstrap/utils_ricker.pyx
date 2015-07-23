#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

from libc.math cimport exp, log, pow
import numpy as np
cimport numpy as np

cpdef double[:, ::1] param_gamma_arr(double r, double sigma, double[::1] n_prevs):
    cdef int dim = n_prevs.shape[0]
    cdef double[::1] alphas = np.empty(dim)
    cdef double[::1] betas = np.empty(dim)
    cdef double[:, ::1] output = np.empty((2, dim))
    cdef double coeff, var
    cdef int i
    var = sigma*sigma
    for i in range(dim):
        coeff = r*n_prevs[i]*exp(-n_prevs[i])
        alphas[i] = 1/var
        betas[i] = 1/alphas[i]*exp(log(coeff) + var/2)
    output[0] = alphas
    output[1] = betas
    return output


cpdef double[::1] func_mean(double[::1] args, int dim, double r):
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        output[i] = log(r) + log(args[i]) - args[i]
    return output

cpdef double func_mean_generalized(double args, double r, double theta):
    return log(r) + log(args) - pow(args, theta)

cpdef double[::1] func_sigma(int dim, double sigma):
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        output[i] = sigma
    return output

cpdef double[::1] func_lam(double[::1] args, int dim, double phi):
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        output[i] = phi*args[i]
    return output

cpdef double[::1] func_shape(double[::1] args, double observation, int dim):
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        output[i] =  args[i] + observation
    return output

cpdef double[::1] func_scale(double[::1] beta, int dim, double phi):
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        output[i] =  beta[i]/(beta[i]*phi+1)
    return output