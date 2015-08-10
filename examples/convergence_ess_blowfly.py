import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.blowflies import BlowflyMap
from bootstrap.abc import ABCFilter
from distributions.distributions2 import Normal

def perform_filter(inits=None, p=6.5, n0=40, sigmap=np.sqrt(0.1), delta=0.16, sigmad=np.sqrt(0.1), tau=14, m=200, NOS=200, NBS=500, observations=None,
                   particle_init=50, mini=50, maxi=1050, step=100):

    if inits == None:
        inits =  np.array([int(Normal().sample(np.array([particle_init], dtype=np.float64), np.array([10], dtype=np.float64))[0])], dtype=np.float64)

    NBS = [i for i in range(mini, maxi, step)]

    Map_blowfly = BlowflyMap(p, n0, sigmap, delta, sigmad, tau, length=NOS, initial=inits)
    observations = Map_blowfly.observations

    initial = {
        'distribution': Normal,
        'shape': particle_init,
        'scale': 10,
    }

    filter = ABCFilter(tau, NOS, NBS, Map_blowfly, m)

    estim_all = []
    likeli_all = []
    ESS_all = []

    count = 0
    for estim, likeli, ESS in filter.filter():
        estim_all.append(estim)
        likeli_all.append(likeli)
        ESS_all.append(ESS)
        count += 1
        print(count)

    output = {
        'observations': observations,
        'estim': estim_all,
        'likeli': likeli_all,
        'ESS': ESS_all,
        'NBS': NBS,
    }

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the bootstrap filter when assessing convergence of the ESS")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--number", type=int, help="Number of observations")
    parser.add_argument("--min", dest="mini", type=int, help="Minimum number of particles")
    parser.add_argument("--max", dest="maxi", type=int, help="Maximum number of particles")
    parser.add_argument("--step", type=int, help="Step size between number of particles")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v}

    output = perform_filter(**arguments)
    number_steps = len(output['NBS'])
    NOBS = len(output['observations']) - 1

    ESS_norm = [[ess/output['NBS'][i] for ess in output['ESS'][i]] for i in range(number_steps)]
    ESS_t = []
    for i in range(30):
        ESS_t.append([ess[i] for ess in ESS_norm])

    for i in range(30):
        plt.ylim((0,1.2))
        plt.plot([i for i in range(number_steps)], ESS_t[i])
        plt.title("Evolution of the ESS for observation %s" % i)
        plt.show()