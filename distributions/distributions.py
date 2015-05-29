import abc
import numpy as np

from math import exp, log
from numba import jit
from scipy.special import gammaln

from utils.utils import underflow

class Distribution(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def sample(self):
        pass

class Normal(Distribution):

    def __init__(self, size=None, func_loc=lambda args: args, func_scale=lambda args: args):
        self.size = size
        self.loc = lambda args: func_loc(*args) if isinstance(args, list) else func_loc(args)
        self.scale = lambda args: func_scale(*args) if isinstance(args, list) else func_scale(args)

    def sample(self, args_loc=0, args_scale=1):
        return np.random.normal(loc=self.loc(args_loc), scale=self.scale(args_scale), size=self.size)

    @underflow
    def density(self, x, args_loc=0, args_scale=1):
        loc = self.loc(args_loc)
        scale = self.scale(args_scale)
        ldensity = -1/2*np.log(2*np.pi) -np.log(scale) - 1/(2*scale**2)*(x-loc)**2
        return np.exp(ldensity)

class LogNormal(Distribution):

    def __init__(self, size=None, func_mean=lambda args: args, func_sigma=lambda args: args):
        self.size = size
        self.mean = lambda args: func_mean(*args) if isinstance(args, list) else func_mean(args)
        self.sigma = lambda args: func_sigma(*args) if isinstance(args, list) else func_sigma(args)

    def sample(self, args_mean=0, args_sigma=1):
        return np.random.lognormal(mean=self.mean(args_mean), sigma=self.sigma(args_sigma), size=self.size)

    @underflow
    def density(self, x, args_mean=0, args_sigma=1):
        mean = self.mean(args_mean)
        sigma = self.sigma(args_sigma)
        ldensity = -np.log(x) - 1/2*np.log(2*np.pi) -np.log(sigma) - 1/(2*sigma**2)*(np.log(x)-mean)**2
        return np.exp(ldensity)

class Poisson(Distribution):

    def __init__(self, size=None, func_lam=lambda args: args):
        self.size = size
        self.lam = lambda args: func_lam(*args) if isinstance(args, list) else func_lam(args)

    def sample(self, args_lam=0):
        return np.random.poisson(lam=self.lam(args_lam), size=self.size)

    @underflow
    def density(self, x, args_lam=0):
        lam = self.lam(args_lam)
        ldensity = -lam + x*np.log(lam) - gammaln(x+1)
        return np.exp(ldensity)

class Gamma(Distribution):

    def __init__(self, size=None, func_shape=lambda args: args, func_scale=lambda args: args):
        self.size = size
        self.shape = lambda args: func_shape(*args) if isinstance(args, list) else func_shape(args)
        self.scale = lambda args: func_scale(*args) if isinstance(args, list) else func_scale(args)

    def sample(self, args_shape=1, args_scale=1):
        return np.random.gamma(shape=self.shape(args_shape), scale=self.scale(args_scale), size=self.size)

    @underflow
    def density(self, x, args_shape=1, args_scale=1):
        shape = self.shape(args_shape)
        scale = self.scale(args_scale)
        ldensity = -shape*np.log(scale) - gammaln(shape) + (shape-1)*np.log(x) - x/scale
        return np.exp(ldensity)

class Uniform(Distribution):

    def __init__(self, size=None, func_low=lambda args: args, func_high=lambda args: args):
        self.size = size
        self.low = lambda args: func_low(*args) if isinstance(args, list) else func_low(args)
        self.high = lambda args: func_high(*args) if isinstance(args, list) else func_high(args)

    def sample(self, args_low=0, args_high=1):
        return np.random.uniform(low=self.low(args_low), high=self.high(args_high), size=self.size)

    @underflow
    def density(self, x, args_low=0, args_high=1):
        low = self.low(args_low)
        high = self.high(args_high)
        return 1/(high-low)

class MultivariateUniform(Distribution):

    def __init__(self, ndims=1, size=None, func_lows=[lambda args: args], func_highs=[lambda args: args]):
        self.ndims = ndims
        self.size = size
        self.lows = [lambda args: func_low(*args) if isinstance(args, list) else func_low(args) for func_low in func_lows]
        self.highs = [lambda args: func_high(*args) if isinstance(args, list) else func_high(args) for func_high in func_highs]

    def sample(self, args_lows=[0], args_highs=[1]):
        samples = []
        for i in range(self.ndims):
            samples.append(np.random.uniform(low=self.lows[i](args_lows[i]), high=self.highs[i](args_highs[i]), size=self.size))
        return samples

    def density(self, args_lows=[0], args_highs=[1]):
        density = 1
        for i in range(self.ndims):
            low = self.lows[i](args_lows[i])
            high = self.highs[i](args_highs[i])
            density *= 1/(high-low)
        return density