import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.blowflies import BlowflyMap
from bootstrap.abc import ABCFilter
from distributions.distributions2 import Normal

def perform_filter(inits=None, p=6.5, n0=40, sigmap=np.sqrt(0.1), delta=0.16, sigmad=np.sqrt(0.1), tau=14, m=100, NOS=100, NBS=200, low=35, high=45,
                   discretization=0.1, observations=None, repeat=10, variable='r',  particle_init=50, tol=0, proposal='optimal'):

    if inits == None:
        inits = np.array([int(Normal().sample(np.array([particle_init], dtype=np.float64), np.array([10], dtype=np.float64))[0])], dtype=np.float64)

    if observations is None:
        Map_ref = BlowflyMap(p, n0, sigmap, delta, sigmad, tau, length=NOS, initial=inits, tol=tol)
        observations = Map_ref.observations
    else:
        observations = observations

    steps = int((high-low)/discretization) + 1
    variables = np.linspace(low, high, steps)
    likelis = []

    mle = []

    for i in range(repeat):
        likeli = -np.inf
        mle_variable = variables[0]
        for variable_ in variables:

            if variable == 'p':
                Map_blowfly = BlowflyMap(variable_, n0, sigmap, delta, sigmad, tau, length=NOS, initial=inits, observations=observations, tol=tol)
            elif variable == 'n0':
                Map_blowfly = BlowflyMap(p, variable_, sigmap, delta, sigmad, tau, length=NOS, initial=inits, observations=observations, tol=tol)
            elif variable == 'delta':
                Map_blowfly = BlowflyMap(p, n0, sigmap, variable_, sigmad, tau, length=NOS, initial=inits, observations=observations, tol=tol)
            elif variable == 'tau':
                Map_blowfly = BlowflyMap(p, n0, sigmap, delta, sigmad, int(variable_), length=NOS, initial=inits, observations=observations, tol=tol)
            elif variable == 'sigmap':
                Map_blowfly = BlowflyMap(p, n0, variable_, delta, sigmad, tau, length=NOS, initial=inits, observations=observations, tol=tol)
            elif variable == 'sigmad':
                Map_blowfly = BlowflyMap(p, n0, sigmap, delta, variable_, tau, length=NOS, initial=inits, observations=observations, tol=tol)
            else:
                raise Exception("Parameter does not exist")

            filter = ABCFilter(NOS, NBS, Map_blowfly, m, likeli=True, proposal={proposal: True})
            _, lik = next(filter.filter())
            if likeli < lik[-1]:
                likeli = lik[-1]
                mle_variable = variable_

        mle.append(mle_variable)

        print(i)

    output = {
        'variable': eval(variable),
        'repeat': repeat,
        'mle': mle
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
    parser.add_argument("--proposal", type=str, help="Proposal distribution: prior or optimal ?")
    parser.add_argument("--tolerance", dest="tol", type=float, help="Tolerance in the bridge")
    parser.add_argument("--repetitions", dest="m", type=int, help="Number of repetitions in inner loop")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'destination'}

    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])

    output = perform_filter(**arguments)

    path = args.destination
    variable = args.variable
    particles = arguments.get('NBS', 500)
    proposal = arguments.get('proposal', 'optimal')
    tolerance = int(arguments.get('tol', 0)*100)
    repetitions = arguments.get('m', 100)
    endpath = "mle_{}_{}_{}_{}_{}.txt".format(variable, particles, proposal, tolerance, repetitions)

    with open(''.join([path, endpath]) if path[-1] == '/' else '/'.join([path, endpath]), 'w') as f:
        f.write(' '.join(map(str, output['mle'])))
        f.write('\n')