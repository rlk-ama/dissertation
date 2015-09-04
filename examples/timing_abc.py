import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

from bootstrap.blowflies import BlowflyMap
from bootstrap.abc import ABCFilter

def perform_filter(inits=None, p=6.5, n0=40, sigmap=np.sqrt(0.1), delta=0.16, sigmad=np.sqrt(0.1), tau=14, m=100, NOS=100, NBS=500, observations=None,
                   particle_init=50, filter_proposal="optimal", tol=0, adaptation=False):

    Map_blowfly = BlowflyMap(p, n0, sigmap, delta, sigmad, tau, length=NOS, initial=inits, observations=observations, tol=tol)

    times = []
    for n in range(10, NBS, 10):
        inner = []
        print(n)
        for i in range(20):
            t1 = time.time()
            filter = ABCFilter(NOS, NBS, Map_blowfly, m, proposal={filter_proposal: True}, adaptation=adaptation)
            _, res = next(filter.filter())
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
    parser.add_argument("--filter_proposal", type=str, help="Proposal for the particle fitler, either prior or optimal")
    parser.add_argument("--repetitions", dest="m", type=int, help="Number of repetitions in inner loop")
    parser.add_argument("--steps", dest="NOS", type=int, help="Number of generations to take into account")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")
    parser.add_argument("--tolerance", dest="tol", type=float, help="Tolerance in the bridge")
    args = parser.parse_args()

    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'destination'}
    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])
    output = perform_filter(**arguments)
    outputs = zip(output['steps'], output['times'])
    particles = arguments.get('NBS', 500)
    proposal = arguments.get('proposal', 'optimal')
    tolerance = int(arguments.get('tol', 0)*100)
    inner = arguments.get('m', 100)
    endpath = "{}_{}_{}_{}".format(particles, proposal, tolerance, inner)
    with open("/home/raphael/timing_blowfly_{}.txt".format(endpath), 'w') as f:
        for elem in outputs:
            f.write(" ".join(map(str, elem)))
            f.write("\n")