import numpy as np

from distributions.distributions2 import LogNormal, Gamma, Poisson, Normal
from utils.utilsc import wrapper_arr, wrapper_arr_arr
from bootstrap.utils_ricker import param_gamma_arr, func_mean, func_lam, func_shape, func_scale, func_sigma, func_mean_generalized
from functools import partial

class RickerMap(object):

    def __init__(self, r, phi, sigma, scaling=1, initial=None, observations=None, length=None, approx='simple',
                 *args, **kwargs):
        self.r = r
        self.phi = phi
        self.sigma = sigma
        self.scaling = scaling

        if initial is not None:
            self.initial = initial
        else:
            self.initial = Normal().sample(loc=np.array([0], dtype=np.float64), scale=np.array([1], dtype=np.float64))

        self.kernel = self.Kernel(LogNormal(), r=self.r, sigma=self.sigma, scaling=self.scaling)
        self.prior = self.kernel
        self.conditional = self.Conditional(Poisson(), phi=self.phi)

        if  approx == 'simple':
            self.proposal = self.Proposal(Gamma(),
                                          self.r, self.sigma, self.phi, self.scaling, param_gamma_arr)
        elif approx == 'complex':
            self.proposal = self.Proposal(Gamma(),
                                          self.r, self.sigma, self.phi,  self.scaling, param_gamma_arr)

        if observations is not None:
            self.observations = observations
        else:
            self.observations, self.state = self.observ_gen(length)

    class Kernel(object):

        def __init__(self, distribution, r, sigma, scaling):
            self.distribution = distribution
            self.r = r
            self.sigma = sigma
            self.scaling = scaling
            self.func_mean = func_mean
            self.func_sigma = func_sigma

        def sample(self, ancestor):
            mean = self.func_mean(ancestor, len(ancestor), self.r, self.scaling)
            sigma = self.func_sigma(len(ancestor), self.sigma)
            return  self.distribution.sample(mean=mean, sigma=sigma)
        #@profile
        def density(self, particle, ancestor):
            mean = self.func_mean(ancestor, len(ancestor), self.r, self.scaling)
            sigma = self.func_sigma(len(ancestor), self.sigma)
            return self.distribution.density(particle, mean=mean, sigma=sigma)

    class Conditional(object):

        def __init__(self, distribution, phi):
            self.distribution = distribution
            self.phi = phi
            self.func_lam = func_lam

        def sample(self, ancestor):
            lam = self.func_lam(ancestor, len(ancestor), self.phi)
            return self.distribution.sample(lam)

        def density(self, particle, observation):
            observations = np.array([observation]*len(particle), dtype=np.float64)
            lam = self.func_lam(particle, len(particle), self.phi)
            return self.distribution.density(observations, lam)

    class Proposal(object):

        def __init__(self, distribution, r, sigma, phi, scaling, param_gamma_arr):
            self.distribution = distribution
            self.r = r
            self.sigma = sigma
            self.phi = phi
            self.scaling = scaling
            self.param_gamma_arr = param_gamma_arr
            self.func_scale = func_scale

        def sample(self, ancestor, observation):
            params = self.param_gamma_arr(self.r, self.sigma, ancestor, self.scaling)
            shape = func_shape(params[0], observation, len(params[0]))
            scale = self.func_scale(params[1], len(params[1]), self.phi)
            return self.distribution.sample(shape=shape, scale=scale)
        #@profile
        def density(self, particle, ancestor, observation):
            params = self.param_gamma_arr(self.r, self.sigma, ancestor, self.scaling)
            shape = func_shape(params[0], observation, len(params[0]))
            scale = self.func_scale(params[1], len(params[1]), self.phi)
            return self.distribution.density(particle, shape=shape, scale=scale)

    def observ_gen(self, length):
        observ = np.empty(length)
        state = np.empty(length)
        x = self.initial
        for i in range(length):
            observ[i] = self.conditional.sample(x)[0]
            state[i] = x[0]
            x = self.kernel.sample(x)
        return observ, state

    #TODO: CHANGE
    def param_gamma2(self, n_prev):

        coeff = self.r*n_prev*np.exp(-n_prev)
        alpha = (6-self.sigma**2)/(3*self.sigma**2)
        beta = 1/alpha*np.exp(np.log(coeff)+ 1/(2*alpha))
        return alpha, beta


class RickerGeneralizedMap(object):

    def __init__(self, r, phi, theta, sigma, initial=None, observations=None, length=None, approx='simple'):
        self.r = r
        self.phi = phi
        self.sigma = sigma
        self.theta = theta

        if initial:
            self.initial = initial
        else:
            self.initial = Normal().sample(loc=np.array([0], dtype=np.float64), scale=np.array([1], dtype=np.float64))

        self.kernel = self.Kernel(LogNormal(), r=self.r, theta=self.theta, sigma=self.sigma)
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

        def __init__(self, distribution, r, theta, sigma):
            self.distribution = distribution
            self.r = r
            self.theta = theta
            self.sigma = sigma
            self.func_mean = partial(func_mean_generalized, r=self.r, theta=self.theta)

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

        def __init__(self, distribution, r, sigma, phi, param_gamma_arr, scaling):
            self.distribution = distribution
            self.r = r
            self.sigma = sigma
            self.phi = phi
            self.scaling = scaling
            self.param_gamma_arr = param_gamma_arr
            self.func_scale = partial(func_scale, phi=self.phi)

        def sample(self, ancestor, observation):
            params = self.param_gamma_arr(self.r, self.sigma, ancestor, self.scaling)
            shape_args = np.array(list(zip(params[0], [observation]*len(ancestor))), dtype=np.float64)
            shape = wrapper_arr_arr(func_shape, shape_args)
            scale = wrapper_arr(self.func_scale, params[1])
            return self.distribution.sample(shape=shape, scale=scale)
        #@profile
        def density(self, particle, ancestor, observation):
            params = self.param_gamma_arr(self.r, self.sigma, ancestor, self.scaling)
            shape_args = np.array(list(zip(params[0], [observation]*len(particle))), dtype=np.float64)
            shape = wrapper_arr_arr(func_shape, shape_args)
            scale = wrapper_arr(self.func_scale, params[1])
            return self.distribution.density(particle, shape=shape, scale=scale)

    def observ_gen(self, length):
        observ = np.empty(length)
        state = np.empty(length)
        x = self.initial
        for i in range(length):
            observ[i] = self.conditional.sample(x)[0]
            state[i] = x[0]
            x = self.kernel.sample(x)
        return observ, state