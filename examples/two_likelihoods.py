import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.ricker_gamma import RickerMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Gamma

def perform_filter(inits=None, r_obs=44.7, phi_obs=10, sigma_obs=0.5, r_esti=46, phi_esti=9, sigma_esti=0.2, NOS=50, NBS=1000):

    if inits == None:
        inits = Gamma().sample(np.array([3], dtype=np.float64), np.array([1], dtype=np.float64))

    Map_obs = RickerMap(r_obs, phi_obs, sigma_obs, length=NOS, initial=inits, approx="simple")
    Map_ricker = RickerMap(r_esti, phi_esti, sigma_esti, length=NOS, initial=inits, approx="simple", observations=Map_obs.observations)
    state = Map_obs.state
    observations = Map_obs.observations

    initial = {
        'distribution': Gamma,
        'shape': 3,
        'scale': 1,
    }

    filter_obs = BootstrapFilter(NOS, NBS, Map_obs, proposal={'optimal': True}, initial=initial)
    filter = BootstrapFilter(NOS, NBS, Map_ricker, proposal={'optimal': True}, initial=initial)
    proposal_obs, estim_obs, likeli_obs, ESS_obs = next(filter_obs.filter())
    proposal, estim, likeli, ESS = next(filter.filter())

    output = {
        'state': state,
        'observations': observations,
        'proposal': proposal,
        'estim': estim,
        'likeli': likeli,
        'likeli_obs': likeli_obs,
        'ESS': ESS,
        'ESS_obs': ESS_obs,
    }

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the bootstrap filter")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--r", nargs=2, type=float,
                        help="Two values for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--phi", nargs=2, type=float,
                        help="Two values for the parameter phi in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--sigma", nargs=2,
                        type=float, help="Two values for the standard deviation of Z_t in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--number", dest="NOS", type=int, help="Number of observations")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'graphics'}

    for key, value in arguments.items():
        if key in {'r', 'phi', 'sigma'}:
            arguments['%s_obs' % key] = value[0]
            arguments['%s_esti' % key] = value[1]

    if 'r' in arguments:
        arguments.pop('r')
    if 'phi' in arguments:
        arguments.pop('phi')
    if 'sigma' in arguments:
        arguments.pop('sigma')

    output = perform_filter(**arguments)

    mean_esti = [np.mean(est) for est in output['estim']]
    NOS = len(mean_esti)

    plt.plot([i for i in range(NOS)], mean_esti)
    plt.plot([i for i in range(NOS)], output['state'])
    plt.title("Simulated state (green) and filtered state (blue)")
    plt.show()

    plt.plot([i for i in range(NOS)], output['ESS_obs'])
    plt.plot([i for i in range(NOS)], output['ESS'])
    plt.title("Effective sample sizes for true (blue) and wrong (green) parameters")
    plt.show()

    plt.plot([i for i in range(NOS)], output['likeli_obs'])
    plt.plot([i for i in range(NOS)], output['likeli'])
    plt.title("Likelihood for true parameters (blue) and wrong parameters (green)")
    plt.show()