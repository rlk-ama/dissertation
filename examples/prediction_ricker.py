import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.ricker_gamma import RickerMap, RickerGeneralizedMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Gamma

def perform_filter(inits=None, r=44.7, phi=10, sigma=0.3, scaling=1, theta=1, NOS=50, NBS=1000, observations=None, generalized=False,
                   filter_proposal='optimal', particle_init=3, rep=10):

    if inits is None:
        inits = Gamma().sample(np.array([3], dtype=np.float64), np.array([1], dtype=np.float64))

    Map_ricker = RickerMap(r, phi, sigma, scaling, length=len(observations), initial=inits, approx="simple", observations=observations)

    observations = Map_ricker.observations
    initial = {
        'distribution': Gamma,
        'shape': particle_init,
        'scale': 1,
    }
    filter = BootstrapFilter(len(observations), NBS, Map_ricker, proposal={filter_proposal: True}, initial=initial)
    proposal, estim, likeli, ESS = next(filter.filter())

    len_simul = len(observations) - NOS
    paths = []
    obs = []
    for i in range(rep):
        Map_ricker = RickerMap(r, phi, sigma, scaling, length=len_simul, initial=np.array([np.mean(estim[NOS])], dtype=np.float64), approx="simple")
        paths.append(Map_ricker.state)
        obs.append(Map_ricker.observations)

    output = {
        'observations': observations,
        'proposal': proposal,
        'estim': estim,
        'likeli': likeli,
        'ESS': ESS,
        'r': r,
        'phi': phi,
        'sigma': sigma,
        'paths': paths,
        'obs': obs
    }

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the bootstrap filter")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--observations", type=argparse.FileType('r'), help="Observations, in a file, space separated", required=True)
    parser.add_argument("--r", type=float, help="Value for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--theta", type=float, help="Value for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}**theta)*exp(-Z_t)")
    parser.add_argument("--phi", type=float, help="Value for the parameter phi in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--sigma", type=float, help="Value for the standard deviation of Z_t in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--scaling", type=float, help="Value for the scaling factor K in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--number", dest="NOS", type=int, help="Number of observations")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")
    parser.add_argument("--particle_init", type=int, help="Mean of initial state value")
    parser.add_argument("--repetitions", dest="rep", type=int, help="Number of predicited paths")
    parser.add_argument("--generalized", type=bool, help="DO you want to use Generalized Ricker Map ?")
    parser.add_argument("--graphics", type=bool, help="Display graphics ?")
    parser.add_argument("--filter_proposal", type=str, help="Proposal for the particle fitler, either prior or optimal")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'graphics'}

    if 'inits' in arguments:
        line = arguments['inits'][0].readline()
        arguments['inits'] = np.array(line.split())

    line = arguments['observations'].readline().split()
    arguments['observations'] = np.array([float(obs) for obs in line])

    output = perform_filter(**arguments)

    mean_esti = [np.mean(est) for est in output['estim']]
    paths = list(zip(*output['paths']))
    mean_predict = [np.mean(path) for path in paths]
    lower_percentile = [np.percentile(path, 2.5) for path in paths]
    higher_percentile = [np.percentile(path, 97.5) for path in paths]

    obs = list(zip(*output['obs']))
    obs_predict = [np.mean(ob) for ob in obs]
    lower_obs = [np.percentile(ob, 2.5) for ob in obs]
    higher_obs = [np.percentile(ob, 97.5) for ob in obs]
    NOS = arguments.get('NOS', 50)

    if args.graphics:
        plt.plot([i for i in range(len(output['observations']))], mean_esti)
        plt.plot([i for i in range(NOS, len(output['observations']))], mean_predict)
        plt.plot([i for i in range(NOS, len(output['observations']))], lower_percentile)
        plt.plot([i for i in range(NOS, len(output['observations']))], higher_percentile)
        plt.title("Original states (green) and states from filtered states (blue")
        plt.show()

        plt.plot([i for i in range(len(output['observations']))], output['observations'])
        plt.plot([i for i in range(NOS, len(output['observations']))], obs_predict)
        plt.plot([i for i in range(NOS, len(output['observations']))], lower_obs)
        plt.plot([i for i in range(NOS, len(output['observations']))], higher_obs)
        plt.title("Original observations (green) and observations from filtered states (blue")
        plt.show()


        # plt.plot([i for i in range(len(output['observations']))], mean_esti)
        # for path in output['paths']:
        #     plt.plot([i for i in range(NOS, len(output['observations']))], path)
        # plt.show()