import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.ricker_gamma import RickerMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Gamma

def perform_filter(inits=None, r=44.7, phi=10, sigma=0.5, NOS=25, NBS=1000, r_low=np.exp(2.5), r_high=np.exp(4.5),
                   discretization=0.5, observations=None, repeat=10):

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

    steps = int((r_high-r_low)/discretization) + 1
    rs = np.linspace(r_low, r_high, steps)

    mse = {'prior': [], 'optimal': []}

    for i in range(repeat):
        likelis = {'prior': -np.inf, 'optimal': -np.inf}
        mse_r = {'prior' : rs[0], 'optimal': rs[0]}
        for r_ in rs:
            Map_ricker = RickerMap(r_, phi, sigma, length=NOS, initial=inits, approx="simple", observations=observations)
            filter = BootstrapFilter(0, NOS, NBS, Map_ricker, proposal={'optimal': True, 'prior': True}, initial=initial, likeli=True)
            for proposal, likeli in filter.filter():
                if likelis[proposal] < likeli[-1]:
                    likelis[proposal] = likeli[-1]
                    mse_r[proposal] = r_

        mse['prior'].append(mse_r['prior'])
        mse['optimal'].append(mse_r['optimal'])

        print(i)

    output = {
        'r': r,
        'repeat': repeat,
        'prior': mse['prior'],
        'optimal': mse['optimal']
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
    parser.add_argument("--r_low", type=float, help="Start value for r parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--r_high", type=float, help="End value for r parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--discretization", type=float, help="Step in discretization of range of values for r parameters in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--repeat", type=int, help="Number of MLE calculations")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v}

    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])

    output = perform_filter(**arguments)



    plt.plot([i for i in range(output['repeat'])], output['prior'])
    plt.axhline(y=output['r'], color='red')
    plt.title("MSE for r with prior proposal")
    plt.show()

    plt.plot([i for i in range(output['repeat'])], output['optimal'])
    plt.axhline(y=output['r'], color='red')
    plt.title("MSE for r with optimal proposal")
    plt.show()

    print(np.var(output['prior']), np.var(output['optimal']))