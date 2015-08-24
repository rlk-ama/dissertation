import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

from bootstrap.ricker_gamma import RickerMap, RickerGeneralizedMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Gamma

def perform_filter(inits=None, r=44.7, phi=10, sigma=0.3, scaling=1, theta=1, NOS=50, NBS=1000, observations=None, generalized=False,
                   filter_proposal='optimal', particle_init=3):

    Map_ricker = RickerMap(r, phi, sigma, scaling, length=NOS, initial=inits, approx="simple", observations=observations)
    initial = {
        'distribution': Gamma,
        'shape': particle_init,
        'scale': 1,
    }

    times = []
    for n in range(10, NBS, 10):
        inner = []
        print(n)
        for i in range(100):
            t1 = time.time()
            filter = BootstrapFilter(NOS, n, Map_ricker, proposal={filter_proposal: True}, initial=initial)
            proposal, estim, likeli, ESS = next(filter.filter())
            t2 = time.time()
            inner.append(t2-t1)
        times.append(np.mean(inner))

    output = {
        'times': times,
        'steps': [n for n in range(10, NBS, 10)]
    }

    return output

if __name__=="__main__":
    parser = argparse.ArgumentParser("Options for the bootstrap filter")
    parser.add_argument("--observations", type=argparse.FileType('r'), help="Observations, in a file, space separated", required=True)
    parser.add_argument("--destination", type=str, help="Path of the destination of the simulations", required=True)
    parser.add_argument("--filter_proposal", type=str, help="Proposal for the particle fitler, either prior or optimal")
    args = parser.parse_args()

    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'destination'}
    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])
    output = perform_filter(**arguments)
    outputs = zip(output['steps'], output['times'])
    with open(args.destination, 'w') as f:
        for elem in outputs:
            f.write(" ".join(map(str, elem)))
            f.write("\n")