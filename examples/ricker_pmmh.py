from bootstrap.filter import BootstrapFilter
from bootstrap.ricker_gamma import RickerMap
from proposals.proposals import RandomWalkProposal, MultivariateUniformProposal
from distributions.distributions2 import Gamma, Normal
from pmcmc.pmmh import PMMH

import argparse
import numpy as np

def simulation(r=44.7, phi=10, sigma=0.3, NOS=50, NBS=500, iter=17500, runs=10, burnin=2500, adaptation=2500,
               r_init=40, phi_init=10, sigma_init=0.2, target=0.18, target_low=0.10, filter_proposal='optimal'):

    sigma_proposals = [5, 0.5, 0.1]
    map_ = RickerMap
    filter = BootstrapFilter

    initial_filter = {
        'distribution': Gamma,
        'shape': 3,
        'scale': 1,
    }

    proposals = [RandomWalkProposal(sigma=sigma_proposal) for sigma_proposal in sigma_proposals]
    prior = MultivariateUniformProposal(ndims=3, func_lows=[lambda args: np.exp(3.2), lambda args: 5, lambda args: 0.15],
                                        func_highs=[lambda args: np.exp(4.17), lambda args: 15, lambda args: 0.45])

    for i in range(runs):
        initial_ricker = Gamma().sample(np.array([3], dtype=np.float64), np.array([1], dtype=np.float64))
        Map = RickerMap(r, phi, sigma, length=NOS, initial=initial_ricker, approx='simple')

        mcmc = PMMH(filter, map_, iter, proposals, prior, [r_init, phi_init, sigma_init], initial_ricker, 0, NOS, NBS, observations=Map.observations,
                    support=lambda x: x[0] > np.exp(3.2)  and x[0] < np.exp(4.1) and x[1] > 5 and x[1] < 15 and x[2] > 0.15 and x[2] < 0.45,
                    adaptation=adaptation, burnin=burnin, target=target, target_low=target_low, initial_filter=initial_filter,
                    filter_proposal=filter_proposal)

        samples, acceptance = mcmc.sample()
        thetas, phis, sigmas = zip(*samples)
        acceptance_rate = np.sum(acceptance[burnin+adaptation:])/len(acceptance[burnin+adaptation:])

        yield thetas, phis, sigmas, acceptance_rate


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the PMMH sampler")
    parser.add_argument("--r", type=float, help="Value for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--phi", type=float, help="Value for the parameter phi in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--sigma", type=float, help="Value for the standard deviation of Z_t in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--number", dest="NOS", type=int, help="Number of observations")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")
    parser.add_argument("--iterations", dest="iter", type=int, help="Number of samples to draw")
    parser.add_argument("--runs", type=int, help="Number of simulations to carry out")
    parser.add_argument("--burnin", type=int, help="Length of the burnin period")
    parser.add_argument("--adaptation", type=int, help="Length of the adaptation period")
    parser.add_argument("--r_init", type=float, help="Start value for r parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--sigma_init", type=float, help="Start value for sigma parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--phi_init", type=float, help="Start value for phi parameters in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--target", type=float, help="Targeted acceptance rate")
    parser.add_argument("--target_low", type=float, help="Minimum acceptance rate")
    parser.add_argument("--filter_proposal", type=str, help="Proposal for the particle fitler, either prior or optimal")
    parser.add_argument("--destination", type=str, help="Path of the destination of the simulations", required=True)

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v is not None and k != 'destination'}
    path = args.destination

    run = 0

    for thetas, phis, sigmas, acceptance_rate in simulation(**arguments):
        with open(''.join([path, 'samples_{}.txt'.format(run)]) if path[-1] == '/' else '/'.join([path, 'samples_{}.txt'.format(run)]), 'w') as f:
            f.write(str(acceptance_rate))
            f.write("\n")
            for j in range(len(thetas)):
                f.write(" ".join(map(str,[thetas[j], phis[j], sigmas[j]])))
                f.write("\n")
        print(run)
        run += 1