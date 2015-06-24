#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

import numpy as np

cimport numpy as np
from cython.parallel import prange

from libc.math cimport lgamma, log, exp, abs
from collections import Iterable

cdef double PI = 3.14159265358979323846

cdef double[::1] density_normal_array(double[::1] x, double[::1] loc, double[::1] scale):
    cdef int dim = x.shape[0]
    cdef double[::1] ldensity = np.empty(dim)
    cdef double[::1] output = np.empty(dim)
    cdef int i
    cdef double var, sigma
    for i in range(dim):
        var = scale[i]*scale[i]
        ldensity[i] = -1/2*log(2*PI) -log(abs(scale[i])) - 1/(2*var)*(x[i]-loc[i])*(x[i]-loc[i])
        output[i] = exp(ldensity[i])
    return output

cdef double[::1] density_lognormal_array(double[::1] x, double[::1] mean, double[::1] sigma, double[::1] output, int dim) nogil:
    cdef int i
    cdef double var
    for i in range(dim):
        var = sigma[i]*sigma[i]
        output[i] = exp(-log(x[i]) - 1/2*log(2*PI) -log(sigma[i]) - 1/(2*var)*(log(x[i])-mean[i])*(log(x[i])-mean[i]))
    return output

cdef double *density_poisson_array(double *x, double *lam, double *output, int dim) nogil:
    cdef int i
    for i in range(dim):
        output[i] = exp(-lam[i] + x[i]*log(lam[i]) - lgamma(x[i]+1))
    return output

cdef double[::1] density_gamma_array(double[::1] x,double[::1] shape, double[::1] scale):
    cdef int dim = x.shape[0]
    cdef double[::1] ldensity = np.empty(dim)
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        ldensity[i] = -shape[i]*log(scale[i]) - lgamma(shape[i]) + (shape[i]-1)*log(x[i]) - x[i]/scale[i]
        output[i] = exp(ldensity[i])
    return output

cdef double[::1] density_uniform_array(double[::1] x, double[::1] low, double[::1] high):
    cdef int dim = x.shape[0]
    cdef double[::1] ldensity = np.empty(dim)
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        if x[i] > high[i] or x[i] < low[i]:
            output[i] = 0
        else:
            output[i] = 1/(high[i]-low[i])
    return output

def wrapper(func, args):
    if isinstance(args, Iterable):
        return func(*args)
    else:
        return func(args)

class Normal(object):

    def __init__(self, size=None):
        self.size = size

    def sample(self, double[::1] loc, double[::1] scale):
        cdef int dim, i
        cdef double[::1] output
        dim = loc.shape[0]
        output = np.empty(dim)
        for i in range(dim):
            output[i] = np.random.normal(loc=loc[i], scale=scale[i], size=self.size)
        return output

    def density(self, double[::1] x, double[::1] loc, double[::1] scale):
        return density_normal_array(x, loc, scale)

class LogNormal(object):

    def __init__(self, size=None):
        self.size = size

    def sample(self, double[::1] mean, double[::1] sigma):
        cdef int dim, i
        cdef double[::1] output
        dim = mean.shape[0]
        output = np.empty(dim)
        for i in range(dim):
            output[i] = np.random.lognormal(mean=mean[i], sigma=sigma[i], size=self.size)
        return output

    def density(self, double[::1] x, double[::1] mean, double[::1] sigma):
        cdef int dim = x.shape[0]
        cdef double[::1] output = np.empty(dim)
        return density_lognormal_array(x, mean, sigma, output, dim)

class Poisson(object):

    def __init__(self, size=None):
        self.size = size

    def sample(self, double[::1] lam):
        cdef int dim, i
        cdef double[::1] output
        dim = lam.shape[0]
        output = np.empty(dim)
        for i in range(dim):
            output[i] = np.random.poisson(lam=lam[i], size=self.size)
        return output

    def density(self, double[::1] x, double[::1] lam):
        cdef int dim = x.shape[0]
        #cdef int[::1] sub_sizes, pos
        cdef double[::1] output = np.empty(dim)
        cdef double *outputPtr =  density_poisson_array(&x[0], &lam[0], &output[0], dim)
        cdef int i, thread
        # sub_sizes = np.zeros(3, np.int32) + dim//3
        # pos = np.zeros(3, np.int32)
        # sub_sizes[3-1] += dim % 3
        # c = np.cumsum(sub_sizes)
        # pos[1], pos[2], pos[3] = c[0], c[1], c[2]
        # for thread in prange(3, nogil=True, chunksize=1, num_threads=3, schedule='static'):
        #     outputPtr = density_poisson_array(&x[pos[thread]], &lam[pos[thread]], &output[pos[thread]], sub_sizes[thread])
        for i in range(dim): #sub_sizes[thread]
            output[i] = outputPtr[i]
        return output

class Gamma(object):

    def __init__(self, size=None):
        self.size = size

    def sample(self, double[::1] shape, double[::1] scale):
        cdef int dim, i
        cdef double[::1] output
        dim = shape.shape[0]
        output  = np.empty(dim)
        for i in range(dim):
            output[i] = np.random.gamma(shape=shape[i], scale=scale[i], size=self.size)
        return output

    def density(self, double[::1] x, double[::1] shape, double[::1] scale):
        return  density_gamma_array(x, shape, scale)

class Uniform(object):

    def __init__(self, size=None):
        self.size = size

    def sample(self, double[::1] low, double[::1] high):
        cdef int dim, i
        cdef double[::1] output
        dim = low.shape[0]
        output = np.empty(dim)
        for i in range(dim):
            output[i] = np.random.uniform(low=low[i], high=high[i], size=self.size)
        return output

    def density(self, double[::1] x, double[::1] low, double[::1] high):
        return density_uniform_array(x, low, high)

class MultivariateUniform(object):

    def __init__(self, ndims=1, size=None, func_lows=[lambda args: args], func_highs=[lambda args: args]):
        self.ndims = ndims
        self.size = size
        self.lows = func_lows
        self.highs = func_highs

    def sample(self, args_lows=[0], args_highs=[1], multi=False):
        cdef int dims = self.ndims
        cdef double[::1] samples = np.empty(dims)
        cdef int i
        cdef double low, high
        for i in range(dims):
            low = wrapper(self.lows[i], args_lows[i])
            high = wrapper(self.highs[i], args_highs[i])
            samples[i] = np.random.uniform(low=low, high=high, size=self.size)
        return samples

    def density(self, xs, args_lows=[0], args_highs=[1], multi=False):
        cdef double low, high, density
        cdef int i, dim
        density = 1
        dim = self.ndims
        for i in range(dim):
            low = wrapper(self.lows[i], args_lows[i])
            high = wrapper(self.highs[i], args_highs[i])
            if xs[i] > high or xs[i] < low:
                density *= 0
            else:
                density *= 1/(high-low)
        return density