from bootstrap.filter import BootstrapFilter
from bootstrap.kalman import KalmanMap
from proposals.proposals import RandomWalkProposal, MultivariateUniformProposal
from distributions.distributions import Normal
from pmcmc.pmmh import PMMH

import numpy as np
import matplotlib.pyplot as plt

phi = 0.9
sigma_state = np.sqrt(0.4)
sigma_obs = np.sqrt(0.6)
mean = 0
sigma = 1
length = 40
NOS = 30
NBS = 50
iter = 10000
sigma_proposals = [0.1, 0.1, 0.1]
inits = [0.3, np.sqrt(0.1), np.sqrt(0.1)]

initial_kalman = Normal(func_loc=lambda args: mean, func_scale=lambda args: sigma)

Map = KalmanMap(phi, sigma_state, sigma_obs, length=length, initial=initial_kalman)
map_ = KalmanMap
filter = BootstrapFilter
proposals = [RandomWalkProposal(sigma=sigma_proposal) for sigma_proposal in sigma_proposals]
prior = MultivariateUniformProposal(ndims=3, func_lows=[lambda args: 0.1, lambda args: -1, lambda args: -1],
                                    func_highs=[lambda args: 2, lambda args: 1, lambda args: 1])

mcmc = PMMH(filter, map_, iter, proposals, prior, inits, initial_kalman, 0, NOS, NBS, observations=Map.observations,
            support=lambda x: x[0] > 0)
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