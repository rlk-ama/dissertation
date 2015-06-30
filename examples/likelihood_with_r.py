import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.ricker_gamma import RickerMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Gamma

def perform_filter(inits=None, r=44.7, phi=10, sigma=0.5, NOS=50, NBS=1000, r_low=np.exp(2.5), r_high=np.exp(4.5), discretization=0.5):

    if inits == None:
        inits = Gamma().sample(np.array([3], dtype=np.float64), np.array([1], dtype=np.float64))

    Map_ref = RickerMap(r, phi, sigma, length=NOS, initial=inits, approx="simple")
    state = Map_ref.state
    observations = Map_ref.observations
    initial = {
        'distribution': Gamma,
        'shape': 3,
        'scale': 1,
    }

    steps = int((r_high-r_low)/discretization) + 1
    rs = np.linspace(r_low, r_high, steps)
    likelis = []

    for r_ in rs:
        Map_ricker = RickerMap(r_, phi, sigma, length=NOS, initial=inits, approx="simple", observations=observations)
        filter = BootstrapFilter(0, NOS, NBS, Map_ricker, proposal={'optimal': True}, initial=initial, likeli=True)
        proposal, likeli = next(filter.filter())
        likelis.append(likeli[-1])

    output = {
        'r': r,
        'state': state,
        'observations': observations,
        'likeli': likelis
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
    parser.add_argument("--r_low", type=float, help="Start value for r parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--r_high", type=float, help="End value for r parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--distretization", type=float, help="Step in discretization of range of values for r parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'graphics'}

    output = perform_filter(**arguments)
    length = len(output['likeli'])

    plt.plot([i for i in range(length)], output['likeli'])
    plt.axvline(x=output['r'])
    plt.title("Likelihood")
    plt.show()
