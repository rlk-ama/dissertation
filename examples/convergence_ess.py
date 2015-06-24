import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.ricker_gamma import RickerMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Gamma

def perform_filter(inits=None, r=44.7, phi=10, sigma=0.5, NOS=50, mini=10, maxi=1000, step=10):

    if inits == None:
        inits = Gamma().sample(np.array([3], dtype=np.float64), np.array([1], dtype=np.float64))

    NBS = [i for i in range(mini, maxi, step)]

    Map_ricker = RickerMap(r, phi, sigma, length=NOS, initial=inits, approx="simple")
    state = Map_ricker.state
    observations = Map_ricker.observations
    initial = {
        'distribution': Gamma,
        'shape': 3,
        'scale': 1,
    }
    filter = BootstrapFilter(0, NOS, NBS, Map_ricker, proposal={'optimal': True}, initial=initial)

    estim_all = []
    likeli_all = []
    ESS_all = []
    proposals_all = []

    for proposal, estim, likeli, ESS in filter.filter():
        proposals_all.append(proposal)
        estim_all.append(estim)
        likeli_all.append(likeli)
        ESS_all.append(ESS)

    output = {
        'state': state,
        'observations': observations,
        'proposal': proposals_all,
        'estim': estim_all,
        'likeli': likeli_all,
        'ESS': ESS_all,
        'NBS': NBS,
    }

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the bootstrap filter when assessing convergence of the ESS")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--r", type=int, help="Value for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--phi", type=float, help="Value for the parameter phi in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--sigma", type=float, help="Value for the standard deviation of Z_t in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--number", type=int, help="Number of observations")
    parser.add_argument("--min", dest="mini", type=int, help="Minimum number of particles")
    parser.add_argument("--max", dest="maxi", type=int, help="Maximum number of particles")
    parser.add_argument("--step", type=int, help="Step size between number of particles")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v}

    output = perform_filter(**arguments)
    number_steps = len(output['NBS'])
    NOBS = len(output['observations']) - 1

    ESS_norm = [[ess/output['NBS'][i] for ess in output['ESS'][i]] for i in range(number_steps)]
    ESS_t = []
    for i in range(NOBS):
        ESS_t.append([ess[i] for ess in ESS_norm])

    for i in range(NOBS):
        plt.ylim((0,1.2))
        plt.plot([i for i in range(number_steps)], ESS_t[i])
        plt.title("Evolution of the ESS for observation %s" % i)
        plt.show()