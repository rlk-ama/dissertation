import numpy as np
import matplotlib.pyplot as plt
import argparse
import pykalman as pk

from bootstrap.kalman import KalmanMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Normal

def perform_filter(inits=None, phi=0.95, scale_state=1, scale_obs=1, NOS=100, NBS=1000, observations=None):

    if inits is None:
        inits = Normal().sample(np.array([0], dtype=np.float64), np.array([1], dtype=np.float64))

    if observations is None:
        Map_kalman = KalmanMap(phi, scale_state, scale_obs, length=NOS, initial=inits)
    else:
        Map_kalman = KalmanMap(phi, scale_state, scale_obs, length=NOS, initial=inits, observations=observations)

    state = None if observations is not None else Map_kalman.state
    observations = Map_kalman.observations

    observations_kalman = observations[:, np.newaxis]
    filter = BootstrapFilter(NOS, NBS, Map_kalman, proposal={'optimal': True}, likeli=True)
    filter_kalman = pk.KalmanFilter(transition_matrices=phi, observation_matrices=1, transition_covariance=scale_state**2,
                                    observation_covariance=scale_obs**2, initial_state_mean=state[0] if state is not None else 0,
                                    initial_state_covariance=scale_state)

    output = {}

    likeli_kalman = filter_kalman.loglikelihood(observations_kalman)
    output['kalman'] = likeli_kalman

    proposal, likeli = next(filter.filter())
    output[proposal] = likeli

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Options for the bootstrap filter")
    parser.add_argument("--inits", type=argparse.FileType('r'), help="Initial value for the state, in a file, space separated")
    parser.add_argument("--observations", type=argparse.FileType('r'), help="Observations, in a file, space separated")
    parser.add_argument("--phi", type=float, help="Value for the parameter phi in the state equation X_t = phi*X_{t-1} + Z_t")
    parser.add_argument("--scale_state", type=float, help="Value for the standard deviation of Z_t in the state equation X_t = phi*X_{t-1} + Z_t")
    parser.add_argument("--scale_obs", type=float, help="Value for the standard deviation of R_t in the observation equation Y_t = X_t + R_t")
    parser.add_argument("--number", dest="NOS", type=int, help="Number of observations")
    parser.add_argument("--particles", dest="NBS", type=int, help="Number of particles")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'inits'}

    if 'inits' in arguments:
        line = arguments['inits'][0].readline()
        arguments['inits'] = np.array(line.split())

    if 'observations' in arguments:
        line = arguments['observations'].readline().split()
        arguments['observations'] = np.array([float(obs) for obs in line])


    output = perform_filter(**arguments)

    print(output['kalman'], output['optimal'][-1])