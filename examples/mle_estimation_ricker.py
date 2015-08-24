import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.ricker_gamma import RickerMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Gamma

def perform_filter(inits=None, r=44.7, phi=10, sigma=0.3, scaling=1, NOS=50, NBS=1000, low=np.exp(2.5), high=np.exp(4.5),
                   discretization=0.5, observations=None, variable='r'):

    if inits == None:
        inits = Gamma().sample(np.array([3], dtype=np.float64), np.array([1], dtype=np.float64))

    if observations is None:
        Map_ref = RickerMap(r, phi, sigma, scaling, length=NOS, initial=inits, approx="simple")
        observations = Map_ref.observations
    else:
        observations = observations

    initial = {
        'distribution': Gamma,
        'shape': 3,
        'scale': 1,
    }

    steps = int((high-low)/discretization) + 1
    variables = np.linspace(low, high, steps)
    likelis = []

    for variable_ in variables:

        if variable == 'r':
            Map_ricker = RickerMap(variable_, phi, sigma, length=NOS, initial=inits, approx="simple", observations=observations)
        elif variable == 'phi':
            Map_ricker = RickerMap(r, variable_, sigma, length=NOS, initial=inits, approx="simple", observations=observations)
        elif variable == 'sigma':
            Map_ricker = RickerMap(r, phi, variable_, length=NOS, initial=inits, approx="simple", observations=observations)
        else:
            raise Exception("Parameter does not exist")

        filter = BootstrapFilter(NOS, NBS, Map_ricker, proposal={'optimal': True}, initial=initial, likeli=True)
        proposal, likeli = next(filter.filter())
        likelis.append(likeli[-1])
        print(variable_)

    output = {
        'variable': r if variable == 'r' else (phi if variable == 'phi' else sigma),
        'variables': variables,
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
    parser.add_argument("--variable", type=str, help="Variable you want to find the MLE of (among r, phi and sigma)", required=True)
    parser.add_argument("--low", type=float, help="Start value for the variable in the state or observation equation")
    parser.add_argument("--high", type=float, help="End value for the variable in the state or observation equation")
    parser.add_argument("--discretization", type=float, help="Step in discretization of range of values for r parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--scaling", type=float, help="Value for the scaling factor K in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")

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

    serial = zip(output['variables'], output['likeli'])
    with open("/home/raphael/mle_{}_ricker.txt".format(arguments['variable']), "w") as f:
        for elem in serial:
            f.write(" ".join(map(str, elem)))
            f.write("\n")
        f.write(" {} {}\n".format(str(output['variable']), str(maxi)))

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
