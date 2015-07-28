#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

from libc.math cimport exp, log, pow, gamma, lgamma
from scipy.integrate import quad
import numpy as np
cimport numpy as np

cdef double func(double x, double particle, double ancestor, double delta, double shape):
    cdef coeff = gamma(ancestor + 1)/(gamma(particle + 1)*gamma(ancestor-particle+1)*pow(shape, shape)/gamma(shape)
    return coeff*exp(-(delta*particle + shape)*x)*pow((1-exp(-delta*x)), ancestor-particle)*pow(x, shape-1)

cpdef double[:: 1] transi_s(double[:: 1] particle, double[:: 1] ancestor, double delta, double shape, int dim):
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        output[i] = quad(func, args=(particle[i], ancestor[i], delta, shape))[0]
    return  output


cpdef double[:: 1] coeff_r(double[:: 1] particle,  double shape, int dim):
    cdef double[:: 1] output = np.empty(dim)
    for i in range(dim):
        output[i] = exp(lgamma(particle[i]+shape) -lgamma(particle[i]+1) -lgamma(shape))
    return output

cpdef double[:: 1] coeff_beta(double[:: 1] ancestor,  double p, double n0, int dim):
    cdef double[:: 1] output = np.empty(dim)
    for i in range(dim):
        output[i] = p*ancestor[i]*exp(-ancestor[i]/n0)
    return output

cpdef double[:: 1] proba_r(double[:: 1] coeff,  double shape, int dim):
    cdef double[:: 1] output = np.empty(dim)
    for i in range(dim):
        output[i] = shape/(coeff[i] + shape)
    return output