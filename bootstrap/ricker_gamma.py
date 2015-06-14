import numpy as np

from distributions.distributions2 import LogNormal, Gamma, Poisson, Normal
from utils.utilsc import param_gamma, param_gamma_arr, func_mean, func_lam, func_shape, func_scale, func_sigma, wrapper_arr, wrapper_arr_arr
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
            self.proposal = self.Proposal(Gamma(func_shape=lambda alpha, obs: alpha + obs,
                                                func_scale=lambda beta: beta/(beta*self.phi+1)),
                                          self.param_gamma2)

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

    # def param_gamma(self, n_prev):
    #     coeff = self.r*n_prev*np.exp(-n_prev)
    #     alpha = 1/self.sigma**2
    #     beta = 1/alpha*np.exp(np.log(coeff)+self.sigma**2/2)
    #     return alpha, beta

    def param_gamma2(self, n_prev):

        coeff = self.r*n_prev*np.exp(-n_prev)
        alpha = (6-self.sigma**2)/(3*self.sigma**2)
        beta = 1/alpha*np.exp(np.log(coeff)+ 1/(2*alpha))
        return alpha, beta


    # def kernel(self, n):
    #     out = np.random.lognormal(mean=0, sigma=self.sigma)
    #     return self.r*n*np.exp(-n)*out
    #
    # def kernel_density(self, n, n_prev):
    #     coeff = self.r*n_prev*np.exp(-n_prev)
    #     mean = np.log(coeff)
    #     out = 1/(n*self.sigma*np.sqrt(2*np.pi))*np.exp(-1/(2*self.sigma**2)*(np.log(n)-mean)**2)
    #     return out


    # def prior_proposal(self, n, obs):
    #     out = np.random.lognormal(mean=0, sigma=self.sigma)
    #     return self.r*n*np.exp(-n)*out
    #
    # def prior_proposal_density(self, n, n_prev, obs):
    #     coeff = self.r*n_prev*np.exp(-n_prev)
    #     mean = np.log(coeff)
    #     out = 1/(n*self.sigma*np.sqrt(2*np.pi))*np.exp(-1/(2*self.sigma**2)*(np.log(n)-mean)**2)
    #     return out


    # def conditional(self, n):
    #     out = np.random.poisson(lam=self.phi*n)
    #     return out
    #
    # def conditional_density(self, n, y):
    #     outlog = -self.phi*n + y*np.log(self.phi*n) - gammaln(y+1)
    #     out = np.exp(outlog)
    #     return out

    # def proposal(self, n, y):
    #     if self.approx == 'simple':
    #         alpha, beta = self.param_gamma(n)
    #     else:
    #         alpha, beta = self.param_gamma2(n)
    #     out = np.random.gamma(shape=y+alpha, scale=beta/(self.phi*beta+1))
    #     return out
    #
    # def proposal_density(self, n, n_prev, y):
    #     if self.approx == 'simple':
    #         alpha, beta = self.param_gamma(n_prev)
    #     else:
    #         alpha, beta = self.param_gamma2(n_prev)
    #     outlog= (alpha+y)*np.log((self.phi*beta+1)/beta) - gammaln(alpha+y) + (alpha+y-1)*np.log(n) - ((self.phi*beta+1)/beta)*n
    #     out = np.exp(outlog)
    #     return  out