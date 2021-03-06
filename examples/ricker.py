import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.ricker_gamma import RickerMap, RickerGeneralizedMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Gamma

def perform_filter(inits=None, r=44.7, phi=10, sigma=0.3, scaling=1, theta=1, NOS=50, NBS=1000, observations=None, generalized=False,
                   filter_proposal='optimal', particle_init=3):

    if inits is None:
        inits = Gamma().sample(np.array([3], dtype=np.float64), np.array([1], dtype=np.float64))

    if observations is None:
        if generalized:
            Map_ricker = RickerGeneralizedMap(r, phi, theta, sigma, length=NOS, initial=inits, approx="simple")
        else:
            Map_ricker = RickerMap(r, phi, sigma, scaling, length=NOS, initial=inits, approx="simple")
    else:
        if generalized:
            Map_ricker = RickerGeneralizedMap(r, phi, theta, sigma, length=NOS, initial=inits, approx="simple", observations=observations)
        else:
            Map_ricker = RickerMap(r, phi, sigma, scaling, length=NOS, initial=inits, approx="simple", observations=observations)

    state = None if observations is not None else Map_ricker.state
    observations = Map_ricker.observations
    initial = {
        'distribution': Gamma,
        'shape': particle_init,
        'scale': 1,
    }
    filter = BootstrapFilter(NOS, NBS, Map_ricker, proposal={filter_proposal: True}, initial=initial)
    proposal, estim, likeli, ESS = next(filter.filter())

    output = {
        'state': state,
        'observations': observations,
        'proposal': proposal,
        'estim': estim,
        'likeli': likeli,
        'ESS': ESS,
        'r': r,
        'phi': phi,
        'sigma': sigma
    }

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the bootstrap filter")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--observations", type=argparse.FileType('r'), help="Observations, in a file, space separated")
    parser.add_argument("--r", type=float, help="Value for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--theta", type=float, help="Value for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}**theta)*exp(-Z_t)")
    parser.add_argument("--phi", type=float, help="Value for the parameter phi in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--sigma", type=float, help="Value for the standard deviation of Z_t in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--scaling", type=float, help="Value for the scaling factor K in the state equation N_t = r*N_{t-1}*exp(-N_{t-1}/K)*exp(-Z_t)")
    parser.add_argument("--number", dest="NOS", type=int, help="Number of observations")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")
    parser.add_argument("--particle_init", type=int, help="Mean of initial state value")
    parser.add_argument("--generalized", type=bool, help="DO you want to use Generalized Ricker Map ?")
    parser.add_argument("--graphics", type=bool, help="Display graphics ?")
    parser.add_argument("--pickle", type=bool, help="Serialize results ?")
    parser.add_argument("--filter_proposal", type=str, help="Proposal for the particle fitler, either prior or optimal")
    parser.add_argument("--stability", type=bool, help="Estimate stability of likelihood estimate")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'graphics' and k != 'pickle' and k != 'stability'}

    if 'inits' in arguments:
        line = arguments['inits'][0].readline()
        arguments['inits'] = np.array(line.split())

    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])

    if args.stability:
        liks = []
        for i in range(100):
            output = perform_filter(**arguments)
            liks.append(output['likeli'][-1])
            print(i)
        with open("/home/raphael/stability_likelihood_ricker_100.txt", "w") as f:
            f.write(" ".join(map(str, liks)))
            f.write("\n")

    else:
        output = perform_filter(**arguments)

        if 'observations' not in arguments:
            with open('/home/raphael/ricker_state.txt', 'w') as f:
                f.write(" ".join(map(str, output['state'])))
                f.write("\n")
            with open('/home/raphael/ricker_obs.txt', 'w') as f:
                f.write(" ".join(map(str, output['observations'])))
                f.write("\n")

        if args.pickle:
            outputs = zip(output['observations'], [np.mean(est) for est in output['estim']], output['likeli'], output['ESS'])
            with open("/home/raphael/pickled_ricker.txt", "w") as f:
                for elem in outputs:
                    f.write(" ".join(map(str, elem)))
                    f.write("\n")

        mean_esti = [np.mean(est) for est in output['estim']]
        NOS = len(mean_esti)

        if args.graphics:
            plt.plot([i for i in range(NOS)], mean_esti if output['state'] is not None else [np.random.poisson(output['phi']*esti) for esti in mean_esti])
            plt.plot([i for i in range(NOS)], output['state'] if output['state'] is not None else output['observations'])
            plt.title("Simulated state (green) and filtered state (blue)" if output['state'] is not None else "Original observations (green) and observations from filtered states (blue")
            plt.show()

            plt.plot([i for i in range(NOS)], output['ESS'])
            plt.ylim(0, arguments['NBS'] + 10 if 'NBS' in arguments else 1010)
            plt.title("Effective sample sizes")
            plt.show()

            plt.plot([i for i in range(NOS)], output['likeli'])
            plt.title("Likelihood")
            plt.show()

            fig, ax1 = plt.subplots()
            ax1.plot([i for i in range(NOS)], output['observations'])
            ax2 = ax1.twinx()
            ax2.plot([i for i in range(NOS)], output['ESS'], color="red")
            plt.title("Observations (red), ESS (blue)")
            plt.show()