import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.kalman import KalmanMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Normal

def perform_filter(inits=None, phi=0.9, scale_state=np.sqrt(0.4), scale_obs=np.sqrt(0.6), NOS=100, NBS=1000):

    if inits == None:
        inits = Normal().sample(np.array([0], dtype=np.float64), np.array([1], dtype=np.float64))

    Map_kalman = KalmanMap(phi, scale_state, scale_obs, length=NOS, initial=inits)
    state = Map_kalman.state
    observations = Map_kalman.observations
    filter = BootstrapFilter(0, NOS, NBS, Map_kalman, proposal={'optimal': True, 'prior': True})

    output_filter = {}
    for proposal, estim, likeli, ESS in filter.filter():
        output_filter[proposal] = {'estim': estim, 'likeli': likeli, 'ESS': ESS}

    output = {
        'state': state,
        'output': output_filter,
        'observation': observations
    }

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the bootstrap filter")
    parser.add_argument("--inits", type=float, help="Initial value for the state")
    parser.add_argument("--phi", type=float, help="Value for the parameter phi in the state equation X_t = phi*X_{t-1} + Z_t")
    parser.add_argument("--scale_state", type=float, help="Value for the standard deviation of Z_t in the state equation X_t = phi*X_{t-1} + Z_t")
    parser.add_argument("--scale_obs", type=float, help="Value for the standard deviation of R_t in the observation equation Y_t = X_t + R_t")
    parser.add_argument("--number", type=int, help="Number of observations")
    parser.add_argument("--particles", type=int, help="Number of particles")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v}

    output = perform_filter(**arguments)

    mean_esti_prior = [np.mean(est) for est in output['output']['prior']['estim']]
    mean_esti_optimal = [np.mean(est) for est in output['output']['optimal']['estim']]
    NOS = len(mean_esti_prior)

    plt.plot([i for i in range(NOS)], mean_esti_prior)
    plt.plot([i for i in range(NOS)], mean_esti_optimal)
    plt.plot([i for i in range(NOS+1)], output['state'])
    plt.title("Simulated state (red) and filtered state obtained with prior proposal (blue) and optimal proposal (green)")
    plt.show()

    plt.plot([i for i in range(NOS)], output['output']['prior']['ESS'])
    plt.plot([i for i in range(NOS)], output['output']['optimal']['ESS'])
    plt.title("Effective sample sizes for the prior proposal (blue) and for the optimal proposal (green)")
    plt.show()

    plt.plot([i for i in range(NOS)], output['output']['prior']['likeli'])
    plt.plot([i for i in range(NOS)], output['output']['optimal']['likeli'])
    plt.title("Likelihood for the prior proposal (blue) and for the optimal proposal (green)")
    plt.show()