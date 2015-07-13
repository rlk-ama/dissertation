import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.ricker_gamma import RickerMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Gamma

def perform_filter(inits=None, r=44.7, phi=10, sigma=0.3, NOS=50, NBS=1000, low=np.exp(2.5), high=np.exp(4.5),
                   discretization=0.5, observations=None, repeat=10, variable='r'):

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

    steps = int((high-low)/discretization) + 1
    variables = np.linspace(low, high, steps)

    mle = {'prior': [], 'optimal': []}

    for i in range(repeat):
        likelis = {'prior': -np.inf, 'optimal': -np.inf}
        mle_variable = {'prior' : variables[0], 'optimal': variables[0]}
        for variable_ in variables:

            if variable == 'r':
                Map_ricker = RickerMap(variable_, phi, sigma, length=NOS, initial=inits, approx="simple", observations=observations)
            elif variable == 'phi':
                Map_ricker = RickerMap(r, variable_, sigma, length=NOS, initial=inits, approx="simple", observations=observations)
            elif variable == 'sigma':
                Map_ricker = RickerMap(r, phi, variable_, length=NOS, initial=inits, approx="simple", observations=observations)
            else:
                raise Exception("Parameter does not exist")

            filter = BootstrapFilter(0, NOS, NBS, Map_ricker, proposal={'optimal': True, 'prior': True}, initial=initial, likeli=True)
            for proposal, likeli in filter.filter():
                if likelis[proposal] < likeli[-1]:
                    likelis[proposal] = likeli[-1]
                    mle_variable[proposal] = variable_

        mle['prior'].append(mle_variable['prior'])
        mle['optimal'].append(mle_variable['optimal'])

        print(i)

    output = {
        'variable': r if variable == 'r' else (phi if variable == 'phi' else sigma),
        'repeat': repeat,
        'prior': mle['prior'],
        'optimal': mle['optimal']
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
    parser.add_argument("--repeat", type=int, help="Number of MLE calculations")
    parser.add_argument("--destination", type=str, help="Destination folder to store results", required=True)

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'destination'}

    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])

    output = perform_filter(**arguments)

    path = args.destination
    variable = args.variable
    particles = args.NBS
    with open(''.join([path, 'prior_{}_{}.txt'.format(variable, particles)]) if path[-1] == '/' else '/'.join([path, 'prior_{}_{}.txt'.format(variable, particles)]), 'w') as f:
        f.write(' '.join(map(str, output['prior'])))

    with open(''.join([path, 'gamma_{}_{}.txt'.format(variable, particles)]) if path[-1] == '/' else '/'.join([path, 'gamma_{}_{}.txt'.format(variable, particles)]), 'w') as f:
        f.write(' '.join(map(str, output['optimal'])))

    plt.plot([i for i in range(output['repeat'])], output['prior'])
    plt.axhline(y=output['variable'], color='red')
    plt.title("MLE for variable with prior proposal")
    plt.show()

    plt.plot([i for i in range(output['repeat'])], output['optimal'])
    plt.axhline(y=output['variable'], color='red')
    plt.title("MLE for variable with optimal proposal")
    plt.show()