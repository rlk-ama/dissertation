import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.kalman import KalmanMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Normal

def perform_filter(inits=None, phi_obs=0.95, scale_state_obs=0.6, scale_obs_obs=0.7,
                   phi_esti=0.7, scale_state_esti=0.1, scale_obs_esti=0.3, NOS=50, NBS=1000):

    if inits is None:
        inits = Normal().sample(np.array([0], dtype=np.float64), np.array([1], dtype=np.float64))

    Map_obs = KalmanMap(phi_obs, scale_state_obs, scale_obs_obs, length=NOS, initial=inits)
    Map_kalman = KalmanMap(phi_esti, scale_state_esti, scale_obs_esti, length=NOS, initial=inits, observations=Map_obs.observations)
    state = Map_obs.state
    observations = Map_obs.observations

    filter_obs = BootstrapFilter(NOS, NBS, Map_obs, proposal={'optimal': True})
    filter = BootstrapFilter(NOS, NBS, Map_kalman, proposal={'optimal': True})
    proposal_obs, estim_obs, likeli_obs, ESS_obs = next(filter_obs.filter())
    proposal, estim, likeli, ESS = next(filter.filter())

    output = {
        'state': state,
        'observations': observations,
        'proposal': proposal,
        'estim': estim,
        'estim_obs': estim_obs,
        'likeli': likeli,
        'likeli_obs': likeli_obs,
        'ESS': ESS,
        'ESS_obs': ESS_obs
    }

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the bootstrap filter")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--phi", nargs=2, type=float,
                        help="Two values for the parameter r in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--scale_state", nargs=2, type=float,
                        help="Two values for the parameter phi in the observation equation Y_t = Poisson(phi*N_t)")
    parser.add_argument("--scale_obs", nargs=2,
                        type=float, help="Two values for the standard deviation of Z_t in the state equation N_t = r*N_{t-1}*exp(-N_{t-1})*exp(-Z_t)")
    parser.add_argument("--number", dest="NOS", type=int, help="Number of observations")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'graphics'}

    for key, value in arguments.items():
        if key in {'phi', 'scale_state', 'scale_obs'}:
            arguments['%s_obs' % key] = value[0]
            arguments['%s_state' % key] = value[1]

    if 'phi' in arguments:
        arguments.pop('phi')
    if 'scale_state' in arguments:
        arguments.pop('scale_state')
    if 'scale_obs' in arguments:
        arguments.pop('scale_obs')

    output = perform_filter(**arguments)

    mean_esti = [np.mean(est) for est in output['estim']]
    mean_esti_obs = [np.mean(est) for est in output['estim_obs']]
    NOS = len(mean_esti)

    plt.plot([i for i in range(NOS)], mean_esti_obs)
    plt.plot([i for i in range(NOS)], mean_esti)
    plt.plot([i for i in range(NOS)], output['state'])
    plt.title("Simulated state (red) and filtered state with true (blue) and wrong (green) parameters")
    plt.show()

    plt.plot([i for i in range(NOS)], output['ESS_obs'])
    plt.plot([i for i in range(NOS)], output['ESS'])
    plt.title("Effective sample sizes for true (blue) and wrong (green) parameters")
    plt.show()

    plt.plot([i for i in range(NOS)], output['likeli_obs'])
    plt.plot([i for i in range(NOS)], output['likeli'])
    plt.title("Likelihood for true parameters (blue) and wrong parameters (wrong)")
    plt.show()