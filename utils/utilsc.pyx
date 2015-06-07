from libc.math cimport exp, pow, log
import numpy as np
cimport numpy as np

cpdef tuple param_gamma(long double r, long double sigma, long double n_prev):
    cdef long double coeff, alpha, beta
    coeff = r*n_prev*exp(-n_prev)
    alpha = 1/pow(sigma, 2)
    beta = 1/alpha*exp(log(coeff) + pow(sigma, 2)/2)
    return alpha, beta

cpdef double[:, ::1] param_gamma_arr(double r, double sigma, double[::1] n_prevs):
    cdef int dim = n_prevs.shape[0]
    cdef double[::1] alphas = np.empty(dim)
    cdef double[::1] betas = np.empty(dim)
    cdef double[:, ::1] output = np.empty((2, dim))
    cdef double coeff
    cdef int i
    for i in range(dim):
        coeff = r*n_prevs[i]*exp(-n_prevs[i])
        alphas[i] = 1/pow(sigma, 2)
        betas[i] = 1/alphas[i]*exp(log(coeff) + pow(sigma, 2)/2)
    output[0] = alphas
    output[1] = betas
    return output