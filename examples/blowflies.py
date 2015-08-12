import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.blowflies import BlowflyMap
from bootstrap.abc import ABCFilter
from distributions.distributions2 import Normal

def perform_filter(inits=None, p=6.5, n0=40, sigmap=np.sqrt(0.1), delta=0.16, sigmad=np.sqrt(0.1), tau=14, m=100, NOS=100, NBS=500, observations=None,
                   particle_init=50, proposal="optimal"):

    if inits is None:
        inits = np.array([int(Normal().sample(np.array([particle_init], dtype=np.float64), np.array([10], dtype=np.float64))[0])], dtype=np.float64)

    if observations is None:
        Map_blowfly = BlowflyMap(p, n0, sigmap, delta, sigmad, tau, length=NOS, initial=inits)
    else:
        Map_blowfly = BlowflyMap(p, n0, sigmap, delta, sigmad, tau, length=NOS, initial=inits, observations=observations)

    observations = Map_blowfly.observations

    initial = {
        'distribution': Normal,
        'shape': particle_init,
        'scale': 10,
    }

    filter = ABCFilter(tau, NOS, NBS, Map_blowfly, m, proposal={proposal: True})
    _, res = next(filter.filter())
    estim, likeli, ESS = res
    output = {
        'observations': observations,
        'estim': estim,
        'likeli': likeli,
        'ESS': ESS
    }

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Options for the bootstrap filter")
    parser.add_argument("--observations", type=argparse.FileType('r'), help="Observations, in a file, space separated")
    parser.add_argument("--proposal", type=str, help="Proposal distribution: prior or optimal ?")
    parser.add_argument("--n0", type=float, help="Value for n0")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v}

    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])

    output = perform_filter(**arguments)

    NOS = len(output['observations'])

    obs_ess_lik = zip(output['observations'], output['ESS'], output['likeli'])
    with open("/home/raphael/abc.txt", "w") as g:
        for elem in obs_ess_lik:
            g.write(" ".join(map(str, elem)))
            g.write("\n")

    if 'observations' not in arguments:
        with open('/home/raphael/blowfly_obs.txt', 'w') as f:
            f.write(" ".join(map(str, output['observations'])))
            f.write("\n")