import numpy as np
import matplotlib.pyplot as plt
import argparse

from bootstrap.kalman import KalmanMap
from bootstrap.filter import BootstrapFilter
from distributions.distributions2 import Normal

def perform_filter(inits=None, phi=0.95, scale_state=1, scale_obs=1, NOS=100, NBS=1000, observations=None,
                   phi_low=0.1, phi_high=1.3, discretization=0.05):

    if inits == None:
        inits = Normal().sample(np.array([0], dtype=np.float64), np.array([1], dtype=np.float64))

    Map_ref = KalmanMap(phi, scale_state, scale_obs, length=NOS, initial=inits)
    state = Map_ref.state
    observations = Map_ref.observations


    steps = int((phi_high-phi_low)/discretization) + 1
    phis = np.linspace(phi_low, phi_high, steps)
    likelis = []

    for phi_ in phis:
        Map_kalman = KalmanMap(phi_, scale_state, scale_obs, length=NOS, initial=inits, observations=observations)
        filter = BootstrapFilter(0, NOS, NBS, Map_kalman, proposal={'optimal': True}, likeli=True)
        proposal, likeli = next(filter.filter())
        likelis.append(likeli[-1])

    output = {
        'phi': phi,
        'phis': phis,
        'state': state,
        'observations': observations,
        'likeli': likelis
    }

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
    parser.add_argument("--phi_low", type=float, help="Start value for phi parameters in the state equation X_t = phi*X_{t-1} + Z_t")
    parser.add_argument("--phi_high", type=float, help="End value for phi parameters in the state equation X_t = phi*X_{t-1} + Z_t")
    parser.add_argument("--distretization", type=float, help="Step in discretization of range of values for phi parameters in the state equation X_t = phi*X_{t-1} + Z_t")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if v and k != 'graphics'}

    output = perform_filter(**arguments)
    length = len(output['likeli'])

    plt.plot(output['phis'], output['likeli'])
    plt.axvline(x=output['phi'])
    plt.title("Likelihood")
    plt.show()