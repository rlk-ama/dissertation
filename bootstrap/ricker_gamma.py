import numpy as np

from distributions.distributions2 import LogNormal, Gamma, Poisson, Normal
from utils.utilsc import wrapper_arr, wrapper_arr_arr
from bootstrap.utils_ricker import param_gamma_arr, func_mean, func_lam, func_shape, func_scale, func_sigma
from functools import partial

class RickerMap(object):

    def __init__(self, r, phi, sigma, initial=None, observations=None, length=None, approx='simple'):
        self.r = r
        self.phi = phi
        self.sigma = sigma

        if initial:
            self.initial = initial
        else:
            self.initial = Normal().sample(loc=np.array([0], dtype=np.float64), scale=np.array([1], dtype=np.float64))

        self.kernel = self.Kernel(LogNormal(), r=self.r, sigma=self.sigma)
        self.prior = self.kernel
        self.conditional = self.Conditional(Poisson(), phi=self.phi)

        if  approx == 'simple':
            self.proposal = self.Proposal(Gamma(),
                                          self.r, self.sigma, self.phi, param_gamma_arr)
        elif approx == 'complex':
            self.proposal = self.Proposal(Gamma(),
                                          self.r, self.sigma, self.phi, param_gamma_arr)

        if observations is not None:
            self.observations = observations
        else:
            self.observations, self.state = self.observ_gen(length)

    class Kernel(object):

        def __init__(self, distribution, r, sigma):
            self.distribution = distribution
            self.r = r
            self.sigma = sigma
            self.func_mean = partial(func_mean, r=self.r)

        def sample(self, ancestor):
            mean = wrapper_arr(self.func_mean, ancestor)
            sigma = wrapper_arr(func_sigma, np.array([self.sigma]*len(ancestor), dtype=np.float64))
            return  self.distribution.sample(mean=mean, sigma=sigma)

        def density(self, particle, ancestor):
            mean = wrapper_arr(self.func_mean, ancestor)
            sigma = wrapper_arr(func_sigma, np.array([self.sigma]*len(ancestor), dtype=np.float64))
            return self.distribution.density(particle, mean=mean, sigma=sigma)

    class Conditional(object):

        def __init__(self, distribution, phi):
            self.distribution = distribution
            self.phi = phi
            self.func_lam = partial(func_lam, phi=self.phi)

        def sample(self, ancestor):
            lam = wrapper_arr(self.func_lam, ancestor)
            return self.distribution.sample(lam)
        #@profile
        def density(self, particle, observation):
            observations = np.array([observation]*len(particle), dtype=np.float64)
            lam = wrapper_arr(self.func_lam, particle)
            return self.distribution.density(observations, lam)

    class Proposal(object):

        def __init__(self, distribution, r, sigma, phi, param_gamma_arr):
            self.distribution = distribution
            self.r = r
            self.sigma = sigma
            self.phi = phi
            self.param_gamma_arr = param_gamma_arr
            self.func_scale = partial(func_scale, phi=self.phi)

        def sample(self, ancestor, observation):
            params = self.param_gamma_arr(self.r, self.sigma, ancestor)
            shape_args = np.array(list(zip(params[0], [observation]*len(ancestor))), dtype=np.float64)
            shape = wrapper_arr_arr(func_shape, shape_args)
            scale = wrapper_arr(self.func_scale, params[1])
            return self.distribution.sample(shape=shape, scale=scale)
        #@profile
        def density(self, particle, ancestor, observation):
            params = self.param_gamma_arr(self.r, self.sigma, ancestor)
            shape_args = np.array(list(zip(params[0], [observation]*len(particle))), dtype=np.float64)
            shape = wrapper_arr_arr(func_shape, shape_args)
            scale = wrapper_arr(self.func_scale, params[1])
            return self.distribution.density(particle, shape=shape, scale=scale)

    def observ_gen(self, length):
        observ = np.empty(length+1)
        state = np.empty(length+1)
        x = self.initial
        for i in range(length+1):
            observ[i] = self.conditional.sample(x)[0]
            state[i] = x[0]
            x = self.kernel.sample(x)
        return observ, state

    #TO DO: CHANGE
    def param_gamma2(self, n_prev):

        coeff = self.r*n_prev*np.exp(-n_prev)
        alpha = (6-self.sigma**2)/(3*self.sigma**2)
        beta = 1/alpha*np.exp(np.log(coeff)+ 1/(2*alpha))
        return alpha, beta
