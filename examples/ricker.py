import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.ricker_gamma import RickerMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Gamma

def perform_filter(inits=None, r=44.7, phi=10, sigma=0.5, NOS=50, NBS=1000):

    if inits == None:
        inits = Gamma().sample(np.array([3], dtype=np.float64), np.array([1], dtype=np.float64))

    Map_ricker = RickerMap(r, phi, sigma, length=NOS, initial=inits, approx="simple")
    state = Map_ricker.state
    observations = Map_ricker.observations
    initial = {
        'distribution': Gamma,
        'shape': 3,
        'scale': 1,
    }
    filter = BootstrapFilter(0, NOS, NBS, Map_ricker, proposal={'optimal': True}, initial=initial)
    proposal, estim, likeli, ESS = next(filter.filter())

    output = {
        'state': state,
        'observations': observations,
        'proposal': proposal,
        'estim': estim,
        'likeli': likeli,
        'ESS': ESS,
    }

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the bootstrap filter")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--r", type=float, help="Value for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--phi", type=float, help="Value for the parameter phi in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--sigma", type=float, help="Value for the standard deviation of Z_t in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--number", type=int, help="Number of observations")
    parser.add_argument("--particles", type=int, help="Number of particles")
    parser.add_argument("--graphics", type=bool, help="Display graphics ?")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'graphics'}

    output = perform_filter(**arguments)

    mean_esti = [np.mean(est) for est in output['estim']]
    NOS = len(mean_esti)

    if args.graphics:
        plt.plot([i for i in range(NOS)], mean_esti)
        plt.plot([i for i in range(NOS+1)], output['state'])
        plt.title("Simulated state (green) and filtered state (blue)")
        plt.show()

        plt.plot([i for i in range(NOS)], output['ESS'])
        plt.title("Effective sample sizes")
        plt.show()

        plt.plot([i for i in range(NOS)], output['likeli'])
        plt.title("Likelihood")
        plt.show()

        fig, ax1 = plt.subplots()
        ax1.plot([i for i in range(NOS+1)], output['observations'])
        ax2 = ax1.twinx()
        ax2.plot([i for i in range(NOS)], output['ESS'], color="red")
        plt.title("Observations (red), ESS (blue)")
        plt.show()