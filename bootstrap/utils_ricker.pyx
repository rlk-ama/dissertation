#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

from libc.math cimport exp, log
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


cpdef double func_mean(double args, double r):
    return log(r) + log(args) - args

cpdef double func_sigma(double sigma):
    return sigma

cpdef double func_lam(double args, double phi):
    return phi*args

cpdef double func_shape(double[::1] args):
    return args[0] + args[1]

cpdef double func_scale(double beta, double phi):
    return beta/(beta*phi+1)