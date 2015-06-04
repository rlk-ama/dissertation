from distributions.distributions import Normal, MultivariateUniform

class RandomWalkProposal(object):

    def __init__(self, sigma=5):
        self.distribution = Normal(func_scale=lambda args: sigma)

    def sample(self, previous):
        return self.distribution.sample(args_loc=previous)

    def density(self, current, previous):
        return self.distribution.density(current, args_loc=previous)

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