from bootstrap.abc import ABCFilter
from bootstrap.blowflies import BlowflyMap
from proposals.proposals import RandomWalkProposal, MultivariateUniformProposal, MultivariateRandomWalkProposal
from distributions.distributions2 import Gamma, Normal
from pmcmc.pmmh import PMMH

import argparse
import numpy as np

def simulation(p=6.5, n0=40, sigmap=np.sqrt(0.1), delta=0.16, sigmad=np.sqrt(0.1), tau=14, m=50, NOS=100, NBS=100,
               iter=1000, chains=1, burnin=0, adaptation=0, p_init=5.5, n0_init=30, delta_init=0.10, sigmap_init=np.sqrt(0.2),
               sigmad_init=np.sqrt(0.2), target=0.44, target_low=0.15, observations=None, inits=None, sigma_proposal_p=0.4,
               sigma_proposal_n0=4, sigma_proposal_delta=0.001, sigma_proposal_sigmap=0.1, sigma_proposal_sigmad=0.1,
               particle_init=50, tol=0.02):

    sigma_proposals = [sigma_proposal_p, sigma_proposal_n0, sigma_proposal_sigmap, sigma_proposal_delta, sigma_proposal_sigmad]
    initial_params = [p_init, n0_init, sigmap_init, delta_init, sigmad_init, 15]
    proposals_bounds = [(1, 60), (1, 1000), (0.1, 3.0), (0.01, 1), (0.1, 3.0), (1, 20)]
    target_mu = [6.5, 40, np.sqrt(0.1), 0.16, np.sqrt(0.1), 14]

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

    cov = np.array([[3.69450361112, -5.04030335342, 0.0914034827708, -0.0128459086697, 0.0887074374203,0],
                    [-5.04030335342, 13.1297421047, -0.143828787466, 0.0417299533945, -0.0935867639508, 0],
                    [0.0914034827708, -0.143828787466, 0.0190265814397, -0.000705206477134, 0.000846422784429, 0],
                    [-0.0128459086697, 0.0417299533945, -0.000705206477134, 0.00146050589699, 8.11165978911e-05, 0],
                    [0.0887074374203, -0.0935867639508, 0.000846422784429, 8.11165978911e-05, 0.0147823231209, 0],
                    [0, 0, 0, 0, 0, 0.3]])

    # cov = np.array([[3.69450361112, -5.04030335342, 0.0914034827708, -0.0128459086697, 0.0887074374203],
    #                 [-5.04030335342, 13.1297421047, -0.143828787466, 0.0417299533945, -0.0935867639508],
    #                 [0.0914034827708, -0.143828787466, 0.0190265814397, -0.000705206477134, 0.000846422784429],
    #                 [-0.0128459086697, 0.0417299533945, -0.000705206477134, 0.00146050589699, 8.11165978911e-05],
    #                 [0.0887074374203, -0.0935867639508, 0.000846422784429, 8.11165978911e-05, 0.0147823231209]])
    #cov = np.diag(sigma_proposals)
    lambdas = [2.38**2/len(initial_params)]*len(initial_params)
    mean = np.array(target_mu)
    proposals = MultivariateRandomWalkProposal(mean=mean, cov=cov, lambdas=lambdas)
    prior = MultivariateUniformProposal(lows=np.array([bound[0] for bound in proposals_bounds]), highs=np.array([bound[1] for bound in proposals_bounds]))
    support = lambda x: all([x[i] > proposals_bounds[i][0] and x[i] < proposals_bounds[i][1] for i in range(len(initial_params))])

    for i in range(chains):

        inits_sampler = proposals.sample(np.array(initial_params))

        while not(support(inits_sampler)):
            inits_sampler = proposals.sample(np.array(initial_params))

        mcmc = PMMH(filter, map_, iter, proposals, prior, inits_sampler, inits, NOS, NBS, observations=observations,
                    support=support, adaptation=adaptation, burnin=burnin, target=target, target_low=target_low,
                    initial_filter=initial_filter, filter_proposal="prior", tol=tol, split=50, rep=m)
        samples, acceptance = mcmc.sample()
        acceptance_rate = np.sum(acceptance[burnin+adaptation:])/(iter-burnin-adaptation)

        yield  samples, acceptance_rate, proposals.cov

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the PMMH sampler")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--observations", type=argparse.FileType('r'), help="Observations, in a file, space separated", required=True)
    parser.add_argument("--number", dest="NOS", type=int, help="Number of observations")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of outer particles")
    parser.add_argument("--inner", dest="m", type=int, help="Number of inner particles")
    parser.add_argument("--particle_init", type=int, help="Mean of initial state value")
    parser.add_argument("--iterations", dest="iter", type=int, help="Number of samples to draw")
    parser.add_argument("--chains", type=int, help="Number of simulations to carry out")
    parser.add_argument("--burnin", type=int, help="Length of the burnin period")
    parser.add_argument("--adaptation", type=int, help="Length of the adaptation period")
    parser.add_argument("--p_init", type=float)
    parser.add_argument("--n0_init", type=float)
    parser.add_argument("--delta_init", type=float)
    parser.add_argument("--sigmap_init", type=float)
    parser.add_argument("--sigmad_init", type=float)
    parser.add_argument("--proposal_p", dest="sigma_proposal_p", type=float)
    parser.add_argument("--proposal_n0", dest="sigma_proposal_n0", type=float)
    parser.add_argument("--proposal_delta", dest="sigma_proposal_delta", type=float)
    parser.add_argument("--proposal_sigmap", dest="sigma_proposal_sigmap", type=float)
    parser.add_argument("--proposal_sigmad", dest="sigma_proposal_sigmad", type=float)
    parser.add_argument("--target", type=float, help="Targeted acceptance rate")
    parser.add_argument("--target_low", type=float, help="Minimum acceptance rate")
    parser.add_argument("--filter_proposal", type=str, help="Proposal for the particle fitler, either prior or optimal")
    parser.add_argument("--destination", type=str, help="Path of the destination of the simulations", required=True)
    parser.add_argument("--tolerance", dest="tol", type=float, help="Tolerance in the bridge")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v is not None and k != 'destination'}
    path = args.destination

    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])

    run = 0

    for samples, acceptance_rate, cov in simulation(**arguments):
        with open(''.join([path, 'samples_{}.txt'.format(run)]) if path[-1] == '/' else '/'.join([path, 'samples_{}.txt'.format(run)]), 'w') as f:
            f.write(str(acceptance_rate))
            f.write("\n")
            f.write(" ".join(map(str, np.ndarray.flatten(cov))))
            f.write("\n")
            for j in range(len(samples)):
                f.write(" ".join(map(str,[parameter for parameter in samples[j]])))
                f.write("\n")
        print(run)
        run += 1