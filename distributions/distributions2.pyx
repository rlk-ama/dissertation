#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

import numpy as np

cimport numpy as np
from cython.parallel import prange

from libc.math cimport lgamma, log, exp
from collections import Iterable
from scipy.special import betaln, beta
from scipy.misc import comb
from scipy.stats import multivariate_normal

cdef double PI = 3.14159265358979323846

cdef double[::1] density_normal_array(double[::1] x, double[::1] loc, double[::1] scale):
    cdef int dim = x.shape[0]
    cdef double[::1] ldensity = np.empty(dim)
    cdef double[::1] output = np.empty(dim)
    cdef int i
    cdef double var, sigma, scale_abs
    for i in range(dim):
        var = scale[i]*scale[i]

        if scale[i] < 0: scale_abs = -scale[i]
        else: scale_abs = scale[i]

        ldensity[i] = -1/2.0*log(2*PI) -log(scale_abs) - 1/(2*var)*(x[i]-loc[i])*(x[i]-loc[i])
        output[i] = exp(ldensity[i])
    return output

cdef double[::1] density_lognormal_array(double[::1] x, double[::1] mean, double[::1] sigma, double[::1] output, int dim) nogil:
    cdef int i
    cdef double var, sigma_abs
    for i in range(dim):
        var = sigma[i]*sigma[i]

        if sigma[i] < 0: sigma_abs = -sigma[i]
        else: sigma_abs = sigma[i]

        output[i] = exp(-log(x[i]) - 1/2.0*log(2*PI) -log(sigma_abs) - 1/(2*var)*(log(x[i])-mean[i])*(log(x[i])-mean[i]))
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

cdef double[::1] density_binomial_array(double[::1] x, double[::1] n, double[::1] p):
    cdef int dim = x.shape[0]
    cdef double[::1] ldensity = np.empty(dim)
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        ldensity[i] = lgamma(n[i]+1) - lgamma(x[i]+1) - lgamma(n[i]+x[i]+1) + x[i]*log(p[i]) + (n[i]-x[i])*log(1-p[i])
        output[i] = exp(ldensity[i])
    return output

cdef double[::1] density_negativebinomial_array(long[::1] x, double[::1] n, double[::1] p):
    cdef int dim = x.shape[0]
    cdef double[::1] ldensity = np.empty(dim)
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        ldensity[i] = lgamma(n[i]+x[i]) - lgamma(x[i]+1) - lgamma(n[i]) + x[i]*log(1-p[i]) + n[i]*log(p[i])
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

class MultivariateNormal(object):


    def sample(self, double[::1] mean, double[:, ::1] cov):
        cdef int dim, i
        cdef double[::1] output
        output = multivariate_normal.rvs(mean=mean, cov=cov, size=1)
        return output

    def density(self, double[::1] x, double[::1] mean, double[:, ::1] cov):
        cdef int dim, i
        cdef double output
        output =  multivariate_normal.pdf(x, mean=mean, cov=cov)
        return output

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


class Binomial(object):

    def __init__(self, size=None):
        self.size = size

    def sample(self, double[::1] n, double[::1] p):
        cdef int dim, i
        cdef double[::1] output
        dim = n.shape[0]
        output  = np.empty(dim)
        for i in range(dim):
            output[i] = np.random.binomial(n=n[i], p=p[i], size=self.size)
        return output

    def density(self, double[::1] x, double[::1] n, double[::1] p):
        return  density_binomial_array(x, n, p)

class BetaBinomial(object):

    def __init__(self, size=None, tweaked=False):
        self.size = size
        self.tweaked = tweaked

    def sample(self, int[::1] n, double[::1] shape1, double[::1] shape2, double[::1] next=None):
        cdef int dim, i, count
        cdef long[::1] output
        cdef long[::1] samples
        cdef double[::1] betas
        cdef long samp
        dim = n.shape[0]
        output = np.zeros(dim, dtype=np.int64)
        i = 0
        betas = np.random.beta(a=shape1, b=shape2)
        samples = np.random.binomial(n=n, p=betas)
        for samp in samples:
            if self.tweaked:
                count = 0
                while samp > next[i]:
                    if count > 100:
                        break
                    beta = np.random.beta(a=shape1[0], b=shape2[0], size=None)
                    samp = np.random.binomial(n=n[i], p=beta, size=None)
                    count += 1
            output[i] = samp
            i += 1
        return output

    def density(self, long[::1] x, int[::1] n, double[::1] shape1, double[::1] shape2):
        cdef int dim, i
        cdef double[::1] output, coeff1, coeff2
        dim = n.shape[0]
        output = np.empty(dim)
        coeff1 = np.empty(dim)
        coeff2 = np.empty(dim)
        for i in range(dim):
            coeff1[i] = x[i] + shape1[i]
            coeff2[i] = n[i] - x[i] + shape2[i]
        output  = comb(n, x)*np.exp(betaln(coeff1, coeff2)-betaln(shape1, shape2))
        return output


class  NegativeBinomial(object):

    def __init__(self, size=None, tweaked=False):
        self.size = size
        self.tweaked = tweaked

    def sample(self, double[::1] n, double[::1] p, double next=0):
        cdef int dim, i, count
        cdef long[::1] samples
        dim = n.shape[0]
        samples = np.random.negative_binomial(n=n, p=p)
        for i in range(dim):
            if self.tweaked:
                count = 0
                while samples[i] > next:
                    if count > 100:
                        break
                    samples[i] = np.random.negative_binomial(n=n[i], p=p[i], size=self.size)
                    count += 1
        return samples

    def density(self, long[::1] x, double[::1] n, double[::1] p):
        return  density_negativebinomial_array(x, n, p)

class MultivariateUniform(object):

    def __init__(self, size=None):
        self.size = size

    def sample(self, double[::1] lows, double[::1] highs):
        cdef int dim
        dim = lows.shape[0]
        cdef double[::1] samples = np.empty(dim)
        cdef int i
        for i in range(dim):
            samples[i] = np.random.uniform(low=lows[i], high=highs[i], size=self.size)
        return samples

    def density(self, double[::1] xs, double[::1] lows, double[::1] highs):
        cdef double density
        cdef int i, dim
        density = 1
        dim = xs.shape[0]
        for i in range(dim):
            if xs[i] > highs[i] or xs[i] < lows[i]:
                density *= 0
            else:
                density *= 1/(highs[i]-lows[i])
        return density