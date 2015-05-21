from bootstrap.filter import BootstrapFilter
from bootstrap.ricker_gamma import RickerMap
from proposals.proposals import RandomWalkProposal, MultivariateUniformProposal
from distributions.distributions import Gamma
from pmcmc.pmmh import PMMH

import numpy as np
import matplotlib.pyplot as plt

r = 44.7
sigma = 0.3
phi = 10
shape = 3
scale = 1
length = 40
NOS = 30
NBS = 100
iter = 100
sigma_proposal = 0.1
inits = [44.7, 10, 0.5]

initial_ricker = Gamma(func_shape=lambda args: shape, func_scale=lambda args: scale)

Map = RickerMap(r, phi, sigma, length=length, initial=initial_ricker, approx='simple')
map = RickerMap
filter = BootstrapFilter
proposal = RandomWalkProposal(sigma=sigma_proposal)
prior = MultivariateUniformProposal(ndims=3, func_lows=[lambda args: np.exp(2), lambda args: 3, lambda args: 0.1],
                                    func_highs=[lambda args: np.exp(4), lambda args: 15, lambda args: 0.7])

mcmc = PMMH(filter, map, iter, proposal, prior, inits, initial_ricker, 0, NOS, NBS, observations=Map.observations)
samples = mcmc.sample()
print('toto')