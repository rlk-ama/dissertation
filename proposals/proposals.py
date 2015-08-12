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

    def __init__(self, lows, highs):
        self.lows = lows
        self.highs = highs
        self.distribution = MultivariateUniform()

    def sample(self):
        return self.distribution.sample(lows=self.lows, highs=self.highs)

    def density(self, xs):
        return self.distribution.density(xs, lows=self.lows, highs=self.highs)

def _iterable(item):
    return np.array([item], dtype=DTYPE)