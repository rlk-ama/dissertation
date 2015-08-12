from bootstrap.abc import ABCFilter
from bootstrap.blowflies import BlowflyMap
from proposals.proposals import RandomWalkProposal, MultivariateUniformProposal
from distributions.distributions2 import Gamma, Normal
from pmcmc.pmmh import PMMH

import argparse
import numpy as np

def simulation(p=6.5, n0=40, sigmap=np.sqrt(0.1), delta=0.16, sigmad=np.sqrt(0.1), tau=14, m=100, NOS=100, NBS=500,
               iter=1000, chains=1, burnin=0, adaptation=0, p_init=6.5, n0_init=40, delta_init=0.2, sigmap_init=np.sqrt(0.1),
               sigmad_init=np.sqrt(0.1), target=0.2, target_low=0.15, observations=None, inits=None, sigma_proposal_p=0.01,
               sigma_proposal_n0=1, sigma_proposal_delta=0.01, sigma_proposal_sigmap=0.01, sigma_proposal_sigmad=0.01,
               particle_init=50):

    sigma_proposals = [sigma_proposal_p, sigma_proposal_n0, sigma_proposal_sigmap, sigma_proposal_delta, sigma_proposal_sigmad]
    initial_params = [p_init, n0_init, sigmap_init, delta_init, sigmad_init ]

    if inits == None:
        inits = Normal().sample(np.array([particle_init], dtype=np.float64), np.array([1], dtype=np.float64))

    if observations is None:
        Map = BlowflyMap(p, n0, sigmap, delta, sigmad, tau, length=NOS, initial=inits)
        observations = Map.observations
    else:
        observations = observations

    map_ = BlowflyMap
    filter = ABCFilter

    initial_filter = {
        'distribution': Normal,
        'shape': particle_init,
        'scale': 1,
    }

    proposals = [RandomWalkProposal(sigma=sigma_proposal) for sigma_proposal in sigma_proposals]
    prior = MultivariateUniformProposal(lows=np.array([param/2 for param in initial_params]), highs=np.array([1.5*param for param in initial_params]))
    support = lambda x: all([x[i] > initial_params[i]/2 and x[i] < 1.5*initial_params[i] for i in range(len(initial_params))])

    for i in range(chains):

        inits_sampler = [Normal().sample(np.array([initial_params[i]], dtype=np.float64),
                                         np.array([sigma_proposals[i]], dtype=np.float64))[0] for i in range(len(initial_params))]

        while not(support(inits_sampler)):
            inits_sampler = [Normal().sample(np.array([initial_params[i]], dtype=np.float64),
                                         np.array([sigma_proposals[i]], dtype=np.float64))[0] for i in range(len(initial_params))]

        mcmc = PMMH(filter, map_, iter, proposals, prior, inits_sampler, inits, tau, NOS, NBS, observations=observations,
                    support=support, adaptation=adaptation, burnin=burnin, target=target, target_low=target_low,
                    initial_filter=initial_filter, filter_proposal="prior")
        samples, acceptance = mcmc.sample()
        acceptance_rate = np.sum(acceptance[burnin+adaptation:])/len(acceptance[burnin+adaptation:])

        yield  samples, acceptance_rate

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the PMMH sampler")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--observations", type=argparse.FileType('r'), help="Observations, in a file, space separated")
    parser.add_argument("--r", type=float, help="Value for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--phi", type=float, help="Value for the parameter phi in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--sigma", type=float, help="Value for the standard deviation of Z_t in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--scaling", type=float, help="Value for the scaling factor K in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--number", dest="NOS", type=int, help="Number of observations")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")
    parser.add_argument("--particle_init", type=int, help="Mean of initial state value")
    parser.add_argument("--iterations", dest="iter", type=int, help="Number of samples to draw")
    parser.add_argument("--chains", type=int, help="Number of simulations to carry out")
    parser.add_argument("--burnin", type=int, help="Length of the burnin period")
    parser.add_argument("--adaptation", type=int, help="Length of the adaptation period")
    parser.add_argument("--r_init", type=float, help="Start value for r parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--sigma_init", type=float, help="Start value for sigma parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--scaling_init", type=float, help="Start value for the scaling factor K in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--phi_init", type=float, help="Start value for phi parameters in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--proposal_r", dest="sigma_proposal_r", type=float, help="Standard deviation for r in the random walk proposal")
    parser.add_argument("--proposal_phi", dest="sigma_proposal_phi", type=float, help="Standard deviation for phi in the random walk proposal")
    parser.add_argument("--proposal_sigma", dest="sigma_proposal_sigma", type=float, help="Standard deviation for sigma in the random walk proposal")
    parser.add_argument("--proposal_scaling", dest="sigma_proposal_scaling", type=float, help="Standard deviation for the scaling factor in the random walk proposal")
    parser.add_argument("--target", type=float, help="Targeted acceptance rate")
    parser.add_argument("--target_low", type=float, help="Minimum acceptance rate")
    parser.add_argument("--filter_proposal", type=str, help="Proposal for the particle fitler, either prior or optimal")
    parser.add_argument("--destination", type=str, help="Path of the destination of the simulations", required=True)
    parser.add_argument("--scaling_model", type=bool, help="Are you using Ricker Map with scaling factor ?")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v is not None and k != 'destination'}
    path = args.destination

    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])

    run = 0

    for samples, acceptance_rate in simulation(**arguments):
        with open(''.join([path, 'samples_{}.txt'.format(run)]) if path[-1] == '/' else '/'.join([path, 'samples_{}.txt'.format(run)]), 'w') as f:
            f.write(str(acceptance_rate))
            f.write("\n")
            for j in range(len(samples)):
                f.write(" ".join(map(str,[parameter for parameter in samples[j]])))
                f.write("\n")
        print(run)
        run += 1