import numpy as np

from distributions.distributions import Normal

class KalmanMap(object):

    def __init__(self, phi, sigma_state, sigma_obs, initial=None, observations=None, length=None):
        self.phi = phi
        self.sigma_state = sigma_state
        self.sigma_obs = sigma_obs
        self.coeff = (self.sigma_obs*self.sigma_state)**2/(self.sigma_obs**2+self.sigma_state**2)
        if initial:
            self.initial = initial
        else:
            self.initial = Normal()

        self.kernel = self.Kernel(Normal(func_loc=lambda args: self.phi*args,
                                            func_scale=lambda args: self.sigma_state))
        self.prior = self.kernel
        self.conditional = self.Conditional(Normal(func_scale=lambda args: self.sigma_obs))
        self.proposal = self.Proposal(Normal(func_loc=lambda x, y: self.coeff*(self.phi*x/self.sigma_state**2 + y/self.sigma_obs**2),
                                             func_scale=lambda args: np.sqrt(self.coeff)))
        if observations:
            self.observations = observations
        else:
            self.observations, self.state = self.observ_gen(length)


    class Kernel(object):

        def __init__(self, distribution):
            self.distribution = distribution

        def sample(self, ancestor):
            return self.distribution.sample(args_loc=ancestor)

        def density(self, particle, ancestor):
            return self.distribution.density(particle, args_loc=ancestor)

    class Conditional(object):

        def __init__(self, distribution):
            self.distribution = distribution

        def sample(self, ancestor):
            return self.distribution.sample(args_loc=ancestor)

        def density(self, particle, observation):
            return self.distribution.density(observation, args_loc=particle)

    class Proposal(object):

        def __init__(self, distribution):
            self.distribution = distribution

        def sample(self, ancestor, observation):
            return self.distribution.sample(args_loc=[ancestor, observation])

        def density(self, particle, ancestor, observation):
            return self.distribution.density(particle, args_loc=[ancestor, observation])

    def observ_gen(self, length):
        observ = []
        state = []
        x = self.initial.sample()
        y = 0
        for i in range(length+1):
            observ.append(self.conditional.sample(x))
            state.append(x)
            x = self.kernel.sample(x)
        return observ, state


    # @staticmethod
    # def normal_density(x, mean, sd):
    #     return 1/(np.sqrt(2*np.pi)*sd)*np.exp(-1/(2*sd**2)*(x - mean)**2)
    #
    # def kernel_density(self, x, x_prev):
    #     mean =self.phi*x_prev
    #     out = KalmanMap.normal_density(x, mean, self.sigma_state)
    #     return out

    # def kernel(self, x):
    #     mean = self.phi*x
    #     return np.random.normal(mean, self.sigma_state)


    # def conditional(self, x):
    #     mean = x
    #     out = np.random.normal(mean, self.sigma_obs)
    #     return out
    #
    # def conditional_density(self, x, y):
    #     mean = x
    #     out = KalmanMap.normal_density(y, mean, self.sigma_obs)
    #     return out

    # def prior_proposal(self, x, y):
    #     return self.kernel(x)
    #
    # def prior_proposal_density(self, x, x_prev, y):
    #     return self.kernel_density(x, x_prev)

    # def proposal(self, x, y):
    #     coeff = (self.sigma_obs*self.sigma_state)**2/(self.sigma_obs**2+self.sigma_state**2)
    #     mean = coeff*(self.phi*x/self.sigma_state**2 + y/self.sigma_obs**2)
    #     sd = np.sqrt(coeff)
    #     return np.random.normal(mean, sd)
    #
    # def proposal_density(self, x, x_prev, y):
    #     coeff = (self.sigma_obs*self.sigma_state)**2/(self.sigma_obs**2+self.sigma_state**2)
    #     mean = coeff*(self.phi*x_prev/self.sigma_state**2 + y/self.sigma_obs**2)
    #     sd = np.sqrt(coeff)
    #     return KalmanMap.normal_density(x, mean, sd)