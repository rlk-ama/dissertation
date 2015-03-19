import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from filter import observ_gen, bootstrap_filter

params = {
    'initial': {'mean': 0.0, 'sd': 1},
    'kernel': {'mean': 0.0, 'sd': 1},
    'proposal': {'mean': 0.0, 'sd': 1},
    'conditional': {'mean': 0.0, 'sd': 1}
    }


def normal_density(x, mean, sd):
    return 1/(np.sqrt(2*np.pi)*sd)*np.exp(-1/(2*sd**2)*(x - mean)**2)

def kernel_density(x, x_prev, param):
    mean = 0.95*x_prev + param['mean']
    out = normal_density(x, mean, param['sd'])
    return out

def kernel(x, param):
    mean = 0.95*x + param['mean']
    return np.random.normal(mean, param['sd'])


def conditional(x, param):
    mean = x + param['mean']
    out = np.random.normal(mean, param['sd'])
    return out

def conditional_density(x, y, param):
    mean = x + param['mean']
    out = normal_density(y, mean, param['sd'])
    return out

def proposal_prior(x, y, param):
    return kernel(x, param)

def proposal_density_prior(x, x_prev, y, param):
    return kernel_density(x, x_prev, param)

def initial(param):
    out = np.random.normal(param['mean'], param['sd'])
    return out


if __name__ == '__main__':
    observ, state = observ_gen(100, params, conditional, initial=initial, kernel=kernel)

    estim, likeli, ESS = bootstrap_filter(param=params, start=0, end=100, N=100, kernel_density=kernel_density, conditional_density=conditional_density,
                           proposal=proposal_prior, proposal_density=proposal_density_prior, initial=initial,
                           observations=observ)

    kf = KalmanFilter(transition_matrices =0.95, observation_matrices =1, transition_covariance=1, observation_covariance=1,
                      initial_state_mean=0, initial_state_covariance=1, n_dim_obs=1, n_dim_state=1)
    measurements = observ
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)


    mean_esti = [np.mean(est) for est in estim]
    fig1 = plt.figure()
    plt.plot([i for i in range(100)], mean_esti)
    plt.plot([i for i in range(101)], filtered_state_means)
    plt.savefig('verif_filter.pdf')
    plt.close()
    fig2 = plt.figure()
    plt.plot([i for i in range(100)], ESS)
    plt.savefig('ESS_filter.pdf')
    plt.close()
    fig3 = plt.figure()
    plt.plot([i for i in range(101)], likeli)
    plt.savefig('likeli_filter.pdf')
    plt.close()