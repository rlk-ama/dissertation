#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

from libc.math cimport exp, log, pow, tgamma, lgamma
from scipy.integrate import quad
import numpy as np
cimport numpy as np

cdef double func(double x, double particle, double ancestor, double delta, double shape):
    return exp(-(delta*particle + shape)*x)*pow((1-exp(-delta*x)), ancestor-particle)*pow(x, shape-1)

cpdef double[:: 1] transi_s(double[:: 1] particles, double ancestor, double delta, double shape, int dim):
    cdef double[::1] output = np.empty(dim)
    cdef double coeff, particle
    cdef int i
    for i in range(dim):
        particle = particles[i]
        coeff = exp(lgamma(ancestor+1)-lgamma(ancestor-particle+1)-lgamma(particle+1)+shape*log(shape)-lgamma(shape))
        inte = quad(func, 0, np.inf, args=(particle, ancestor, delta, shape))[0]
        output[i] = exp(log(coeff) + log(inte))
    return  output


cpdef double[:: 1] coeff_r(double[:: 1] particle,  double shape, int dim):
    cdef double[:: 1] output = np.empty(dim)
    for i in range(dim):
        output[i] = exp(lgamma(particle[i]+shape) -lgamma(particle[i]+1) -lgamma(shape))
    return output

cpdef double[:: 1] coeff_beta(double ancestor,  double p, double n0, int dim):
    cdef double[:: 1] output = np.empty(dim)
    for i in range(dim):
        output[i] = p*ancestor*exp(-ancestor/n0)
    return output

cpdef double[:: 1] proba_r(double[:: 1] coeff,  double shape, int dim):
    cdef double[:: 1] output = np.empty(dim)
    for i in range(dim):
        output[i] = shape/(coeff[i] + shape)
    return output