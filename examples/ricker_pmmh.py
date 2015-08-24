from bootstrap.filter import BootstrapFilter
from bootstrap.ricker_gamma import RickerMap, RickerGeneralizedMap
from proposals.proposals import RandomWalkProposal, MultivariateUniformProposal
from distributions.distributions2 import Gamma, Normal
from pmcmc.pmmh import PMMH

import argparse
import numpy as np

def simulation(r=44.7, phi=10, theta=1, sigma=0.3, NOS=50, NBS=500, iter=17500, runs=10, burnin=2500, adaptation=2500,
               r_init=40, phi_init=10, theta_init=0.9, sigma_init=0.4, target=0.15, target_low=0.10, filter_proposal='optimal', generalized=False):

    if generalized:
        sigma_proposals = [5, 0.5, 0.05, 0.1]
        initial_values = [r_init, phi_init, theta_init, sigma_init]
        map_ = RickerGeneralizedMap
    else:
        sigma_proposals = [5, 0.5, 0.1]
        initial_values = [r_init, phi_init, sigma_init]
        map_ = RickerMap
        proposals_bounds = [(1, 60), (1, 50), (0.1, 1.2)]

    filter = BootstrapFilter

    initial_filter = {
        'distribution': Gamma,
        'shape': 3,
        'scale': 1,
    }

    proposals = [RandomWalkProposal(sigma=sigma_proposal) for sigma_proposal in sigma_proposals]


    prior = MultivariateUniformProposal(lows=np.array([bound[0] for bound in proposals_bounds]), highs=np.array([bound[1] for bound in proposals_bounds]))
    support = lambda x: all([x[i] > proposals_bounds[i][0] and x[i] < proposals_bounds[i][1] for i in range(len(initial_values))])
    #prior = MultivariateUniformProposal(lows=np.array([param/2 for param in initial_values]), highs=np.array([1.5*param for param in initial_values]))
    #support = lambda x: all([x[i] > initial_values[i]/2 and x[i] < 1.5*initial_values[i] for i in range(len(initial_values))])

    for i in range(runs):
        initial_ricker = Gamma().sample(np.array([3], dtype=np.float64), np.array([1], dtype=np.float64))

        if generalized:
            Map = RickerGeneralizedMap(r, phi, theta, sigma, length=NOS, initial=initial_ricker, approx='simple')
        else:
            Map = RickerMap(r, phi, sigma, length=NOS, initial=initial_ricker, approx='simple')

        if len(filter_proposal) > 1:
            mcmc = PMMH(filter, map_, iter, proposals, prior, initial_values, initial_ricker, NOS, NBS, observations=Map.observations,
                        support=support, adaptation=adaptation, burnin=burnin, target=target, target_low=target_low, initial_filter=initial_filter,
                        filter_proposal=filter_proposal[0])
            mcmc2 = PMMH(filter, map_, iter, proposals, prior, initial_values, initial_ricker, NOS, int(NBS*1.05), observations=Map.observations,
                        support=support, adaptation=adaptation, burnin=burnin, target=target, target_low=target_low, initial_filter=initial_filter,
                        filter_proposal=filter_proposal[1])

            samples, acceptance = mcmc2.sample()

            if generalized:
                rs, phis, thetas, sigmas = zip(*samples)
            else:
                rs, phis, sigmas = zip(*samples)
            acceptance_rate = np.sum(acceptance[burnin+adaptation:])/len(acceptance[burnin+adaptation:])

            if generalized:
                yield filter_proposal[1], rs, phis, thetas, sigmas, acceptance_rate
            else:
                yield filter_proposal[1], rs, phis, sigmas, acceptance_rate

        else:
            mcmc = PMMH(filter, map_, iter, proposals, prior, initial_values, initial_ricker, NOS, NBS, observations=Map.observations,
                        support=support, adaptation=adaptation, burnin=burnin, target=target, target_low=target_low, initial_filter=initial_filter,
                        filter_proposal=filter_proposal[0])

        samples, acceptance = mcmc.sample()

        if generalized:
            rs, phis, thetas, sigmas = zip(*samples)
        else:
            rs, phis, sigmas = zip(*samples)
        acceptance_rate = np.sum(acceptance[burnin+adaptation:])/len(acceptance[burnin+adaptation:])

        if generalized:
            yield filter_proposal[0], rs, phis, thetas, sigmas, acceptance_rate
        else:
            yield filter_proposal[0], rs, phis, sigmas, acceptance_rate


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
    parser.add_argument("--filter_proposal", type=str, nargs='+', help="Proposal for the particle fitler, either prior or optimal")
    parser.add_argument("--destination", type=str, help="Path of the destination of the simulations", required=True)
    parser.add_argument("--generalized", type=bool, help="Are you using the Generalized Ricker map ?")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v is not None and k != 'destination'}
    path = args.destination

    run = 20

    for proposal, *variables, acceptance_rate in simulation(**arguments):
        with open(''.join([path, 'samples_{}_{}.txt'.format(run, proposal)]) if path[-1] == '/' else '/'.join([path, 'samples_{}_{}.txt'.format(run, proposal)]), 'w') as f:
            f.write(str(acceptance_rate))
            f.write("\n")
            for j in range(len(variables[0])):
                f.write(" ".join(map(str, [variable[j] for variable in variables])))
                f.write("\n")
        print(run)
        run += 1