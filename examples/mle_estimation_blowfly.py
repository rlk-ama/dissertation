import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.blowflies import BlowflyMap
from bootstrap.abc import ABCFilter
from distributions.distributions2 import Normal

def perform_filter(inits=None, p=6.5, n0=40, sigmap=np.sqrt(0.1), delta=0.16, sigmad=np.sqrt(0.1), tau=14, m=100, NOS=200, NBS=500, observations=None,
                   particle_init=50, low=35, high=45, discretization=0.5, variable='n0'):

    if inits == None:
        inits = np.array([int(Normal().sample(np.array([particle_init], dtype=np.float64), np.array([10], dtype=np.float64))[0])], dtype=np.float64)

    if observations is None:
        Map_ref = BlowflyMap(p, n0, sigmap, delta, sigmad, tau, length=NOS, initial=inits)
        observations = Map_ref.observations
    else:
        observations = observations

    steps = int((high-low)/discretization) + 1
    variables = np.linspace(low, high, steps)
    likelis = []

    for variable_ in variables:

        if variable == 'p':
            Map_blowfly = BlowflyMap(variable_, n0, sigmap, delta, sigmad, tau, length=NOS, initial=inits, observations=observations)
        elif variable == 'n0':
            Map_blowfly = BlowflyMap(p, variable_, sigmap, delta, sigmad, tau, length=NOS, initial=inits, observations=observations)
        elif variable == 'delta':
            Map_blowfly = BlowflyMap(p, n0, sigmap, variable_, sigmad, tau, length=NOS, initial=inits, observations=observations)
        else:
            raise Exception("Parameter does not exist")

        filter = ABCFilter(tau, NOS, NBS, Map_blowfly, m, likeli=True)
        likeli = next(filter.filter())
        likelis.append(likeli[-1])
        print(variable_)

    output = {
        'variable': p if variable == 'p' else (n0 if variable == 'n0' else delta),
        'variables': variables,
        'observations': observations,
        'likeli': likelis
    }

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the ABC filter")
    parser.add_argument("--observations", type=argparse.FileType('r'), help="Observations, in a file, space separated")
    parser.add_argument("--number", dest="NOS", type=int, help="Number of observations")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")
    parser.add_argument("--variable", type=str, help="Variable you want to find the MLE of (among p, n0 and delta)", required=True)
    parser.add_argument("--low", type=float, help="Start value for the variable in the state or observation equation")
    parser.add_argument("--high", type=float, help="End value for the variable in the state or observation equation")
    parser.add_argument("--discretization", type=float, help="Step in discretization of range of values for the parameter")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v}

    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])

    output = perform_filter(**arguments)
    length = len(output['likeli'])

    maxi_variable = max(output['likeli'])
    maxi_idx = output['likeli'].index(maxi_variable)
    maxi  = output['variables'][maxi_idx]

    plt.plot(output['variables'], output['likeli'])
    plt.axvline(x=output['variable'])
    plt.axvline(x=maxi, color='red')
    plt.title("Likelihood")
    plt.show()

    plt.plot(output['variables'], output['likeli'])
    plt.axvline(x=output['variable'])
    plt.axvline(x=maxi, color='red')
    plt.title("Likelihood on log scale")
    plt.xscale('log')
    plt.show()