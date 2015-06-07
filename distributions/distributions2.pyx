import numpy as np

cimport numpy as np
cimport cython
from cython.view cimport array as cvarray

from libc.math cimport lgamma, log, exp, pow
from collections import Iterable

cdef double PI = 3.14159265358979323846

#ctypedef double (*func_s)(double)
#ctypedef double (*func_c)(double[::1])

cdef double density_normal(double x, double loc, double scale):
    cdef double ldensity, output, var
    var = scale*scale
    ldensity = -1/2*log(2*PI) -log(scale) - 1/(2*var)*(x-loc)*(x-loc)
    output = exp(ldensity)
    return output

cdef double[::1] density_normal_array(double[::1] x, double[::1] loc, double[::1] scale):
    cdef int dim = x.shape[0]
    cdef double[::1] ldensity = np.empty(dim)
    cdef double[::1] output = np.empty(dim)
    cdef int i
    cdef double var
    for i in range(dim):
        var = scale[i]*scale[i]
        ldensity[i] = -1/2*log(2*PI) -log(scale[i]) - 1/(2*var)*(x[i]-loc[i])*(x[i]-loc[i])
        output[i] = exp(ldensity[i])
    return output

cdef double density_lognormal(double x, double mean, double sigma):
    cdef double ldensity, output, var
    var = sigma*sigma
    ldensity = -log(x) - 1/2*log(2*PI) -log(sigma) - 1/(2*var)*(log(x)-mean)*(log(x)-mean)
    output = exp(ldensity)
    return output

cdef double[::1] density_lognormal_array(double[::1] x, double[::1] mean, double[::1] sigma):
    cdef int dim = x.shape[0]
    cdef double[::1] ldensity = np.empty(dim)
    cdef double[::1] output = np.empty(dim)
    cdef int i
    cdef double var
    for i in range(dim):
        var = sigma[i]*sigma[i]
        ldensity[i] = -log(x[i]) - 1/2*log(2*PI) -log(sigma[i]) - 1/(2*var)*(log(x[i])-mean[i])*(log(x[i])-mean[i])
        output[i] = exp(ldensity[i])
    return output

cdef double density_poisson(double x, double lam):
    cdef double ldensity, output
    ldensity = -lam + x*log(lam) - lgamma(x+1)
    output = exp(ldensity)
    return output

cdef double[::1] density_poisson_array(double[::1] x, double[::1] lam):
    cdef int dim = x.shape[0]
    cdef double[::1] ldensity = np.empty(dim)
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        ldensity[i] = -lam[i] + x[i]*log(lam[i]) - lgamma(x[i]+1)
        output[i] = exp(ldensity[i])
    return output

cdef double density_gamma(double x, double shape, double scale):
    cdef double ldensity, output
    ldensity = -shape*log(scale) - lgamma(shape) + (shape-1)*log(x) - x/scale
    output = exp(ldensity)
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

cdef double density_uniform(double x, double low, double high):
    cdef double output
    if x > high or x < low:
        output = 0
    else:
        output = 1/(high-low)
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

def wrapper_arr(func, args):
    cdef int dim = args.shape[0]
    cdef double[::1] output = np.empty(dim)
    cdef int i
    if isinstance(args[0], Iterable):
        for i in range(dim):
            output[i] = func(*args[i])
    else:
        for i in range(len(args)):
            output[i] = func(args[i])
    return output

# class Distribution(metaclass=abc.ABCMeta):
#
#     @abc.abstractmethod
#     def sample(self):
#         pass

class Normal(object):

    def __init__(self, size=None, func_loc=lambda args: args, func_scale=lambda args: args):
        self.size = size
        self.loc = func_loc
        self.scale = func_scale

    def sample(self, args_loc=0, args_scale=1, multi=False):
        cdef int dim, i
        cdef double[::1] output
        if multi:
            dim = args_loc.shape[0]
            output = np.empty(dim)
            loc = wrapper_arr(self.loc, args_loc)
            scale = wrapper_arr(self.scale, args_scale)
            for i in range(dim):
                output[i] = np.random.normal(loc=loc[i], scale=scale[i], size=self.size)
            return output
        else:
            return np.random.normal(loc=self.loc(args_loc), scale=self.scale(args_scale), size=self.size)

    #@underflow
    def density(self, x, args_loc=0, args_scale=1, multi=False):
        cdef double loc, scale
        cdef double[::1] loc_arr, scale_arr
        if multi:
            loc_arr = wrapper_arr(self.loc, args_loc)
            scale_arr = wrapper_arr(self.scale, args_scale)
            return density_normal_array(x, loc_arr, scale_arr)
        else:
            loc = wrapper(self.loc, args_loc)
            scale =  wrapper(self.scale, args_loc)
            return density_normal(x, loc, scale)

class LogNormal(object):

    def __init__(self, size=None, func_mean=lambda args: args, func_sigma=lambda args: args):
        self.size = size
        self.mean =func_mean
        self.sigma = func_sigma

    def sample(self, args_mean=0, args_sigma=1, multi=False):
        cdef int dim, i
        cdef double[::1] output
        if multi:
            dim = args_mean.shape[0]
            output = np.empty(dim)
            mean = wrapper_arr(self.mean, args_mean)
            sigma = wrapper_arr(self.sigma, args_sigma)
            for i in range(dim):
                output[i] = np.random.lognormal(mean=mean[i], sigma=sigma[i], size=self.size)
            return output
        else:
            return np.random.lognormal(mean=self.mean(args_mean), sigma=self.sigma(args_sigma), size=self.size)

    #@underflow
    def density(self, x, args_mean=0, args_sigma=1, multi=False):
        cdef double mean, sigma
        cdef double[::1] mean_arr, sigma_arr
        if multi:
            mean_arr = wrapper_arr(self.mean, args_mean)
            sigma_arr = wrapper_arr(self.sigma, args_sigma)
            return density_lognormal_array(x, mean_arr, sigma_arr)
        else:
            mean = wrapper(self.mean, args_mean)
            sigma = wrapper(self.sigma, args_sigma)
            return density_lognormal(x, mean, sigma)

class Poisson(object):

    def __init__(self, size=None, func_lam=lambda args: args):
        self.size = size
        self.lam = func_lam

    def sample(self, args_lam=0, multi=False):
        cdef int dim, i
        cdef double[::1] output
        if multi:
            dim = args_lam.shape[0]
            output = np.empty(dim)
            lam = wrapper_arr(self.lam, args_lam)
            for i in range(dim):
                output[i] = np.random.poisson(lam=lam[i], size=self.size)
            return output
        else:
            return np.random.poisson(lam=self.lam(args_lam), size=self.size)

    #@underflow
    def density(self, x, args_lam=0, multi=False):
        cdef double lam, x_d
        cdef double[::1] lam_arr
        if multi:
            lam_arr = wrapper_arr(self.lam, args_lam)
            return density_poisson_array(x, lam_arr)
        else:
            lam = wrapper(self.lam, args_lam)
            return density_poisson(x, lam)

class Gamma(object):

    def __init__(self, size=None, func_shape=lambda args: args, func_scale=lambda args: args):
        self.size = size
        self.shape = func_shape
        self.scale = func_scale

    def sample(self, args_shape=1, args_scale=1, multi=False):
        cdef int dim, i
        cdef double[::1] output
        if multi:
            dim = args_shape.shape[0]
            output  = np.empty(dim)
            shape = wrapper_arr(self.shape, args_shape)
            scale = wrapper_arr(self.scale, args_scale)
            for i in range(dim):
                output[i] = np.random.gamma(shape=shape[i], scale=scale[i], size=self.size)
            return output
        else:
            shape = wrapper(self.shape, args_shape)
            scale = wrapper(self.scale, args_scale)
            return np.random.gamma(shape=shape, scale=scale, size=self.size)

    #@underflow
    def density(self, x, args_shape=1, args_scale=1, multi=False):
        cdef double shape, scale
        cdef double[::1] shape_arr, scale_arr
        if multi:
            shape_arr = wrapper_arr(self.shape, args_shape)
            scale_arr = wrapper_arr(self.scale, args_scale)
            return  density_gamma_array(x, shape_arr, scale_arr)
        else:
            shape = wrapper(self.shape, args_shape)
            scale = wrapper(self.scale, args_scale)
            return  density_gamma(x, shape, scale)

class Uniform(object):

    def __init__(self, size=None, func_low=lambda args: args, func_high=lambda args: args):
        self.size = size
        self.low = func_low
        self.high = func_high

    def sample(self, args_low=0, args_high=1):
        return np.random.uniform(low=self.low(args_low), high=self.high(args_high), size=self.size)

    #@underflow
    def density(self, x, args_low=0, args_high=1):
        cdef double low, high
        if isinstance(x, np.ndarray):
            low_arr = wrapper_arr(self.low, args_low)
            high_arr = wrapper_arr(self.high, args_high)
            return density_uniform_array(x, low_arr, high_arr)
        else:
            low = wrapper(self.low, args_low)
            high = wrapper(self.low, args_low)
            return density_uniform(x, low, high)

class MultivariateUniform(object):

    def __init__(self, ndims=1, size=None, func_lows=[lambda args: args], func_highs=[lambda args: args]):
        self.ndims = ndims
        self.size = size
        self.lows = [lambda args, func=func_low: (func(*args) if isinstance(args, Iterable) else func(args)) for func_low in func_lows]
        self.highs = [lambda args, func=func_high: func(*args) if isinstance(args, Iterable) else func(args) for func_high in func_highs]

    def sample(self, args_lows=[0], args_highs=[1]):
        samples = []
        for i in range(self.ndims):
            samples.append(np.random.uniform(low=self.lows[i](args_lows[i]), high=self.highs[i](args_highs[i]), size=self.size))
        return samples

    def density(self, xs, args_lows=[0], args_highs=[1]):
        density = 1
        for i in range(self.ndims):
            low = self.lows[i](args_lows[i])
            high = self.highs[i](args_highs[i])
            if xs[i] > high or xs[i] < low:
                density *= 0
            else:
                density *= 1/(high-low)
        return density