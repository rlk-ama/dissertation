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
    cdef double coeff = tgamma(ancestor + 1)/(tgamma(particle + 1)*tgamma(ancestor-particle+1))*pow(shape, shape)/tgamma(shape)
    return coeff*exp(-(delta*particle + shape)*x)*pow((1-exp(-delta*x)), ancestor-particle)*pow(x, shape-1)

cpdef double[:: 1] transi_s(double[:: 1] particles, int ancestor, double delta, double shape, int dim):
    cdef double[::1] output = np.empty(dim)
    cdef double internal_sum
    cdef int i, j, particle, sgn
    for i in range(dim):
        internal_sum = 0
        particle = int(particles[i])

        for j in range(ancestor- particle):
            if j % 2 == 0:
                sgn = 1
            else:
                sgn = -1

            internal_sum = internal_sum + exp(-lgamma(j+1) - lgamma(ancestor-particle-j+1) - shape*log(delta*(particle+j)+shape))*sgn

        output[i] = exp(lgamma(ancestor+1)-lgamma(particle+1))*internal_sum
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