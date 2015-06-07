import numpy as np

from distributions.distributions2 import LogNormal, Gamma, Poisson, Normal
from utils.utilsc import param_gamma, param_gamma_arr
from collections import Iterable

class RickerMap(object):

    def __init__(self, r, phi, sigma, initial=None, observations=None, length=None, approx='simple'):
        self.r = r
        self.phi = phi
        self.sigma = sigma
        if initial:
            self.initial = initial
        else:
            self.initial = Normal()
            # self.initial = partial(np.random.normal, {'loc': 0, 'scale': 1000})
        #self.approx = approx

        self.kernel = self.Kernel(LogNormal(func_mean=lambda args: np.log(self.r) + np.log(args) - args,
                                            func_sigma=lambda args: self.sigma))
        self.prior = self.kernel
        self.conditional = self.Conditional(Poisson(func_lam=lambda args: self.phi*args))
        if  approx == 'simple':
            self.proposal = self.Proposal(Gamma(func_shape=lambda alpha, obs: alpha + obs,
                                                func_scale=lambda beta: beta/(beta*self.phi+1)),
                                          self.r, self.sigma, param_gamma, param_gamma_arr)
        elif approx == 'complex':
            self.proposal = self.Proposal(Gamma(func_shape=lambda alpha, obs: alpha + obs,
                                                func_scale=lambda beta: beta/(beta*self.phi+1)),
                                          self.param_gamma2)

        if observations:
            self.observations = observations
        else:
            self.observations, self.state = self.observ_gen(length)

    class Kernel(object):

        def __init__(self, distribution):
            self.distribution = distribution

        def sample(self, ancestor, multi=False):
            return self.distribution.sample(args_mean=ancestor, multi=multi)

        def density(self, particle, ancestor):
            sigma = np.array([1]*len(particle), dtype=np.float64)
            return self.distribution.density(particle, args_mean=ancestor, args_sigma=sigma,multi=True)

    class Conditional(object):

        def __init__(self, distribution):
            self.distribution = distribution

        def sample(self, ancestor, multi=False):
            return self.distribution.sample(args_lam=ancestor, multi=multi)

        def density(self, particle, observation):
            observations = np.array([observation]*len(particle), dtype=np.float64)
            return self.distribution.density(observations, args_lam=particle, multi=True)

    class Proposal(object):

        def __init__(self, distribution, r, sigma, param_gamma, param_gamma_arr):
            self.distribution = distribution
            self.r = r
            self.sigma = sigma
            self.param_gamma = param_gamma
            self.param_gamma_arr = param_gamma_arr

        def sample(self, ancestor, observation):
            #if isinstance(ancestor, np.ndarray):
            params = self.param_gamma_arr(self.r, self.sigma, ancestor)
            shape = np.array(list(zip(params[0], [observation]*len(ancestor))), dtype=np.float64)
            return self.distribution.sample(args_shape=shape, args_scale=params[1], multi=True)
            # else:
            #     alpha, beta = self.param_gamma(self.r, self.sigma, ancestor)
            #     return self.distribution.sample(args_shape=[alpha, observation], args_scale=beta)

        def density(self, particle, ancestor, observation):
            #if isinstance(particle, np.ndarray):
            params = self.param_gamma_arr(self.r, self.sigma, ancestor)
            shape = np.array(list(zip(params[0], [observation]*len(particle))), dtype=np.float64)
            return self.distribution.density(particle, args_shape=shape, args_scale=params[1], multi=True)
            # else:
            #     alpha, beta = self.param_gamma(self.r, self.sigma, ancestor)
            #     return self.distribution.density(particle, args_shape=[alpha, observation], args_scale=beta)

    def observ_gen(self, length):
        observ = np.empty(length+1)
        state = np.empty(length+1)
        x = self.initial.sample()
        y = 0
        for i in range(length+1):
            observ[i] = self.conditional.sample(x)
            state[i] = x
            x = self.kernel.sample(x)
        return observ, state

    def param_gamma(self, n_prev):
        coeff = self.r*n_prev*np.exp(-n_prev)
        alpha = 1/self.sigma**2
        beta = 1/alpha*np.exp(np.log(coeff)+self.sigma**2/2)
        return alpha, beta

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