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
    filter = BootstrapFilter(NOS, NBS, Map_ricker, proposal={'optimal': True, 'prior': True}, initial=initial)

    output_filter = {}
    for proposal, estim, likeli, ESS in filter.filter():
        output_filter[proposal] = {'estim': estim, 'likeli': likeli, 'ESS': ESS}

    output = {
        'state': state,
        'observations': observations,
        'output': output_filter
    }

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the bootstrap filter")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--r", type=float, help="Value for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--phi", type=float, help="Value for the parameter phi in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--sigma", type=float, help="Value for the standard deviation of Z_t in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--number", dest="NOS", type=int, help="Number of observations")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v}

    output = perform_filter(**arguments)
    NOS = len(output['output']['prior']['estim'])

    plt.plot([i for i in range(NOS)], output['output']['prior']['likeli'])
    plt.plot([i for i in range(NOS)], output['output']['optimal']['likeli'])
    plt.title("Likelihood for prior (blue) and optimal proposal (green)")
    plt.show()
