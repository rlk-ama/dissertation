import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.ricker_gamma import RickerMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Gamma

def perform_filter(inits=None, r=44.7, phi=10, sigma=0.5, NOS=50, NBS=1000, sigma_low=0.3, sigma_high=0.8,
                   discretization=0.01, observations=None):

    if inits == None:
        inits = Gamma().sample(np.array([3], dtype=np.float64), np.array([1], dtype=np.float64))

    if observations is None:
        Map_ref = RickerMap(r, phi, sigma, length=NOS, initial=inits, approx="simple")
        observations = Map_ref.observations
    else:
        observations = observations

    initial = {
        'distribution': Gamma,
        'shape': 3,
        'scale': 1,
    }

    steps = int((sigma_high-sigma_low)/discretization) + 1
    sigmas = np.linspace(sigma_low, sigma_high, steps)
    likelis = []

    for sigma_ in sigmas:
        Map_ricker = RickerMap(r, phi, sigma_, length=NOS, initial=inits, approx="simple", observations=observations)
        filter = BootstrapFilter(0, NOS, NBS, Map_ricker, proposal={'optimal': True}, initial=initial, likeli=True)
        proposal, likeli = next(filter.filter())
        likelis.append(likeli[-1])

    output = {
        'sigma': sigma,
        'sigmas': sigmas,
        'observations': observations,
        'likeli': likelis
    }

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the bootstrap filter")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--observations", type=argparse.FileType('r'), help="Observations, in a file, space separated")
    parser.add_argument("--r", type=float, help="Value for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--phi", type=float, help="Value for the parameter phi in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--sigma", type=float, help="Value for the standard deviation of Z_t in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--number", dest="NOS", type=int, help="Number of observations")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")
    parser.add_argument("--sigma_low", type=float, help="Start value for sigma parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--sigma_high", type=float, help="End value for simga parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--discretization", type=float, help="Step in discretization of range of values for r parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'graphics'}

    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])

    output = perform_filter(**arguments)
    length = len(output['likeli'])

    maxi_sigma = max(output['likeli'])
    maxi_idx = output['likeli'].index(maxi_sigma)
    maxi  = output['sigmas'][maxi_idx]

    plt.plot(output['sigmas'], output['likeli'])
    plt.axvline(x=output['sigma'])
    plt.axvline(x=maxi, color='red')
    plt.title("Likelihood")
    plt.show()
