import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.blowflies import BlowflyMap
from bootstrap.abc import ABCFilter
from distributions.distributions2 import Normal

def perform_filter(inits=None, p=6.5, n0=40, sigmap=np.sqrt(0.1), delta=0.16, sigmad=np.sqrt(0.1), tau=14, m=100, NOS=100, NBS=500, observations=None,
                   particle_init=50, proposal="optimal", tol=0, adaptation=False):

    if inits is None:
        inits = np.array([int(Normal().sample(np.array([particle_init], dtype=np.float64), np.array([10], dtype=np.float64))[0])], dtype=np.float64)

    if observations is None:
        Map_blowfly = BlowflyMap(p, n0, sigmap, delta, sigmad, tau, length=NOS, initial=inits, tol=tol)
    else:
        Map_blowfly = BlowflyMap(p, n0, sigmap, delta, sigmad, tau, length=NOS, initial=inits, observations=observations, tol=tol)

    observations = Map_blowfly.observations

    filter = ABCFilter(NOS, NBS, Map_blowfly, m, proposal={proposal: True}, adaptation=adaptation)
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
    parser.add_argument("--delta", type=float, help="Value for delta")
    parser.add_argument("--p", type=float, help="Value for p")
    parser.add_argument("--tau", type=int, help="Value for tau")
    parser.add_argument("--sigmap", type=float, help="Value for sigmap")
    parser.add_argument("--sigmad", type=float, help="Value for sigmad")
    parser.add_argument("--repetitions", dest="m", type=int, help="Number of repetitions in inner loop")
    parser.add_argument("--steps", dest="NOS", type=int, help="Number of generations to take into account")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")
    parser.add_argument("--tolerance", dest="tol", type=float, help="Tolerance in the bridge")
    parser.add_argument("--adaptation", type=bool, help="Use adaptation for number of inner samples ?")
    parser.add_argument("--stability", type=bool, help="Estimate stability of likelihood estimate")


    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'stability'}

    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])

    if args.stability:
        liks = []
        for i in range(15):
            output = perform_filter(**arguments)
            liks.append(output['likeli'][-1])
            print(i)

        particles = arguments.get('NBS', 500)
        proposal = arguments.get('proposal', 'optimal')
        tolerance = int(arguments.get('tol', 0)*100)
        inner = arguments.get('m', 100)
        endpath = "{}_{}_{}_{}".format(particles, proposal, tolerance, inner)
        with open("/home/raphael/stability_likelihood_blowfly_{}.txt".format(endpath), "w") as f:
            f.write(" ".join(map(str, liks)))
            f.write("\n")

    else:
        output = perform_filter(**arguments)

        NOS = len(output['observations'])

        obs_ess_lik = zip(output['observations'], output['ESS'], output['likeli'])
        with open("/home/raphael/abc_{}_{}_{}.txt".format(arguments.get('m', 100),
                                                          arguments.get('NBS', 500),
                                                          arguments.get('tol', 0)), "w") as g:
            for elem in obs_ess_lik:
                g.write(" ".join(map(str, elem)))
                g.write("\n")

        if 'observations' not in arguments:
            with open('/home/raphael/blowfly2_obs.txt', 'w') as f:
                f.write(" ".join(map(str, output['observations'])))
                f.write("\n")