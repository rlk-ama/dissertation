from bootstrap.filter import BootstrapFilter
from bootstrap.ricker_gamma import RickerMap
from proposals.proposals import RandomWalkProposal, MultivariateUniformProposal, MultivariateRandomWalkProposal
from distributions.distributions2 import Gamma, Normal
from pmcmc.pmmh import PMMH

import argparse
import numpy as np

def simulation(r=44.7, phi=10, sigma=0.3, scaling=1, NOS=50, NBS=500, iter=17500, chains=10, burnin=2500, adaptation=2500,
               r_init=40, phi_init=10, sigma_init=0.3, scaling_init=1, target=0.2, target_low=0.15, observations=None, inits=None,
               filter_proposal='optimal', sigma_proposal_r=5, sigma_proposal_phi=0.5, sigma_proposal_sigma=0.1, sigma_proposal_scaling=0,
               scaling_model=False, particle_init=3):

    if scaling_model:
        sigma_proposals = [sigma_proposal_r, sigma_proposal_phi, sigma_proposal_sigma, sigma_proposal_scaling]
        initial_params = [r_init, phi_init, sigma_init, scaling_init]
        target_mu = [40, 10, 0.3, 1]
        proposals_bounds = [(1, 60), (1, 50), (0.01, 1.2), (1, 10000)]
    else:
        sigma_proposals = [sigma_proposal_r, sigma_proposal_phi, sigma_proposal_sigma]
        initial_params = [r_init, phi_init, sigma_init]
        target_mu = [40, 10, 0.3]
        proposals_bounds = [(1, 60), (1, 50), (0.1, 1.2)]

    if inits == None:
        inits = Gamma().sample(np.array([particle_init], dtype=np.float64), np.array([1], dtype=np.float64))

    if observations is None:
        Map = RickerMap(r, phi, sigma, scaling, length=NOS, initial=inits, approx="simple")
        observations = Map.observations
    else:
        observations = observations

    map_ = RickerMap
    filter = BootstrapFilter

    initial_filter = {
        'distribution': Gamma,
        'shape': particle_init,
        'scale': 1,
    }

    cov = np.diag(sigma_proposals)
    lambdas = [2.38**2/len(initial_params)]*len(initial_params)
    mean = np.array(target_mu)
    proposals = MultivariateRandomWalkProposal(mean=mean, cov=cov, lambdas=lambdas)
    prior = MultivariateUniformProposal(lows=np.array([bound[0] for bound in proposals_bounds]), highs=np.array([bound[1] for bound in proposals_bounds]))
    support = lambda x: all([x[i] > proposals_bounds[i][0] and x[i] < proposals_bounds[i][1] for i in range(len(initial_params))])
    #prior = MultivariateUniformProposal(lows=np.array([param/2 for param in initial_params]), highs=np.array([1.5*param for param in initial_params]))
    #support = lambda x: all([x[i] > initial_params[i]/2 and x[i] < 1.5*initial_params[i] for i in range(len(initial_params))])

    for i in range(chains):

        inits_sampler = proposals.sample(np.array(initial_params))

        while not(support(inits_sampler)):
            inits_sampler = proposals.sample(np.array(initial_params))

        mcmc = PMMH(filter, map_, iter, proposals, prior, inits_sampler, inits, NOS, NBS, observations=observations,
                    support=support, adaptation=adaptation, burnin=burnin, target=target, target_low=target_low, initial_filter=initial_filter,
                    filter_proposal=filter_proposal)
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