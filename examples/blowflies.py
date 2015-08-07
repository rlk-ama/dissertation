import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.blowflies import BlowflyMap
from bootstrap.abc import ABCFilter
from distributions.distributions2 import Normal

def perform_filter(inits=None, p=6.5, n0=400, sigmap=np.sqrt(0.1), delta=0.16, sigmad=np.sqrt(0.1), tau=14, m=100, NOS=200, NBS=1000, observations=None,
                   filter_proposal='optimal', particle_init=1000):

    if inits is None:
        inits = np.array([int(Normal().sample(np.array([1000], dtype=np.float64), np.array([10], dtype=np.float64))[0])], dtype=np.float64)

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
    filter = ABCFilter(tau, NOS, NBS, Map_blowfly, m)
    estim, likeli, ESS = next(filter.filter())

    output = {
        'observations': observations,
        'estim': estim,
        'likeli': likeli,
        'ESS': ESS
    }

    return output

if __name__ == "__main__":
    output = perform_filter()

    NOS = len(output['observations'])

    ess_lik = zip(output['ESS'], output['likeli'])
    with open("/home/raphael/abc.txt", "w") as g:
        for elem in ess_lik:
            g.write(" ".join(map(str, elem)))
            g.write("\n")