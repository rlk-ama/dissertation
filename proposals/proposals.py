from distributions.distributions2 import Normal, MultivariateUniform
from collections import Iterable
import numpy as np

DTYPE = np.float64

class RandomWalkProposal(object):

    def __init__(self, sigma=5):
        self.sigma = sigma
        self.distribution = Normal()

    def sample(self, previous):
        if not isinstance(self.sigma, Iterable):
            self.sigma = _iterable(self.sigma)
        if not isinstance(previous, Iterable):
            previous = _iterable(previous)
        return self.distribution.sample(loc=previous, scale=self.sigma)

    def density(self, current, previous):
        if not isinstance(self.sigma, Iterable):
            self.sigma = _iterable(self.sigma)
        if not isinstance(current, Iterable):
            current = _iterable(current)
        if not isinstance(previous, Iterable):
            previous = _iterable(previous)
        return self.distribution.density(current, loc=previous, scale=self.sigma)

class MultivariateUniformProposal(object):

    def __init__(self, ndims=1, func_lows=[lambda args: args], func_highs=[lambda args: args]):
        self.ndims = ndims
        self.distribution = MultivariateUniform(ndims=ndims, func_lows=func_lows, func_highs=func_highs)

    def sample(self, lows=[0], highs=[1]):
        lows = lows*self.ndims
        highs = highs*self.ndims
        return self.distribution.sample(args_lows=lows, args_highs=highs)

    def density(self, xs, lows=[0], highs=[1]):
        lows = lows*self.ndims
        highs = highs*self.ndims
        return self.distribution.density(xs, args_lows=lows, args_highs=highs)

def _iterable(item):
    return np.array([item], dtype=DTYPE)