from bootstrap.filter import BootstrapFilter
from bootstrap.ricker_gamma import RickerMap
from proposals.proposals import RandomWalkProposal, MultivariateUniformProposal
from distributions.distributions import Gamma
from pmcmc.pmmh import PMMH

import numpy as np
import matplotlib.pyplot as plt

r = 20
sigma = 0.3
phi = 10
shape = 3
scale = 1
length = 40
NOS = 30
NBS = 200
iter = 5000
sigma_proposals = [2, 0.5, 0.1]
inits = [44.7, 10, 0.5]

initial_ricker = Gamma(func_shape=lambda args: shape, func_scale=lambda args: scale)

Map = RickerMap(r, phi, sigma, length=length, initial=initial_ricker, approx='simple')
map_ = RickerMap
filter = BootstrapFilter
proposals = [RandomWalkProposal(sigma=sigma_proposal) for sigma_proposal in sigma_proposals]
prior = MultivariateUniformProposal(ndims=3, func_lows=[lambda args: np.exp(2), lambda args: 3, lambda args: 0.1],
                                    func_highs=[lambda args: np.exp(4), lambda args: 15, lambda args: 0.7])

mcmc = PMMH(filter, map_, iter, proposals, prior, inits, initial_ricker, 0, NOS, NBS, observations=Map.observations,
            support=lambda x: x[0] > 0 and x[1] > 0)
samples, acceptance = mcmc.sample()
thetas, phis, sigmas = zip(*samples)
print(np.sum(acceptance)/len(acceptance))
plt.plot([i for i in range(iter+1)], thetas)
plt.show()
plt.plot([i for i in range(iter+1)], phis)
plt.show()
plt.plot([i for i in range(iter+1)], sigmas)
plt.show()
hist, bins = np.histogram(thetas, bins=40, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
plt.show()
hist, bins = np.histogram(phis, bins=40, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
plt.show()
hist, bins = np.histogram(sigmas, bins=40, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
plt.show()