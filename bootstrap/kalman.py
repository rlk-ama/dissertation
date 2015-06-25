import numpy as np

from distributions.distributions2 import Normal
from utils.utilsc import wrapper_arr, wrapper_arr_arr
from bootstrap.utils_kalman import func_loc_state, func_scale_state, func_scale_obs, func_loc_proposal, func_scale_proposal
from functools import partial

class KalmanMap(object):

    def __init__(self, phi, scale_state, scale_obs, initial=None, observations=None, length=None):
        self.phi = phi
        self.scale_state = scale_state
        self.scale_obs = scale_obs
        self.scale = (self.scale_obs*self.scale_state)**2/(self.scale_obs**2+self.scale_state**2)

        if initial:
            self.initial = initial
        else:
            self.initial = Normal().sample(loc=np.array([0], dtype=np.float64), scale=np.array([1], dtype=np.float64))

        self.kernel = self.Kernel(Normal(), self.phi, self.scale_state)
        self.prior = self.kernel
        self.conditional = self.Conditional(Normal(), self.scale_obs)
        self.proposal = self.Proposal(Normal(), self.phi, self.scale_state, self.scale_obs, self.scale)

        if observations is not None:
            self.observations = observations
        else:
            self.observations, self.state = self.observ_gen(length)


    class Kernel(object):

        def __init__(self, distribution, phi, scale):
            self.distribution = distribution
            self.phi = phi
            self.scale = scale
            self.func_loc = partial(func_loc_state, self.phi)

        def sample(self, ancestor):
            loc = wrapper_arr(self.func_loc, ancestor)
            scale = wrapper_arr(func_scale_state, np.array([self.scale]*len(ancestor), dtype=np.float64))
            return self.distribution.sample(loc=loc, scale=scale)

        def density(self, particle, ancestor):
            loc = wrapper_arr(self.func_loc, ancestor)
            scale = wrapper_arr(func_scale_state, np.array([self.scale]*len(ancestor), dtype=np.float64))
            return self.distribution.density(particle, loc=loc, scale=scale)

    class Conditional(object):

        def __init__(self, distribution, scale):
            self.distribution = distribution
            self.scale = scale

        def sample(self, ancestor):
            scale = wrapper_arr(func_scale_obs, np.array([self.scale]*len(ancestor), dtype=np.float64))
            return self.distribution.sample(loc=ancestor, scale=scale)

        def density(self, particle, observation):
            observations = np.array([observation]*len(particle), dtype=np.float64)
            scale = wrapper_arr(func_scale_obs, np.array([self.scale]*len(particle), dtype=np.float64))
            return self.distribution.density(observations, loc=particle, scale=scale)

    class Proposal(object):

        def __init__(self, distribution, phi, scale_state, scale_obs, scale):
            self.distribution = distribution
            self.phi = phi
            self.scale_state= scale_state
            self.scale_obs = scale_obs
            self.scale = scale
            self.func_loc = partial(func_loc_proposal, phi=self.phi, scale_state=self.scale_state, scale_obs=self.scale_obs,
                                    scale=self.scale)

        def sample(self, ancestor, observation):
            loc_args = np.array(list(zip(ancestor, [observation]*len(ancestor))), dtype=np.float64)
            loc = wrapper_arr_arr(self.func_loc, loc_args)
            scale = wrapper_arr(func_scale_proposal, np.array([self.scale]*len(ancestor), dtype=np.float64))
            return self.distribution.sample(loc=loc, scale=scale)

        def density(self, particle, ancestor, observation):
            loc_args = np.array(list(zip(ancestor, [observation]*len(ancestor))), dtype=np.float64)
            loc = wrapper_arr_arr(self.func_loc, loc_args)
            scale = wrapper_arr(func_scale_proposal, np.array([self.scale]*len(ancestor), dtype=np.float64))
            return self.distribution.density(particle, loc=loc, scale=scale)

    def observ_gen(self, length):
        observ = np.empty(length)
        state = np.empty(length)
        x = self.initial
        for i in range(length):
            observ[i] = self.conditional.sample(x)[0]
            state[i] = x[0]
            x = self.kernel.sample(x)
        return observ, state