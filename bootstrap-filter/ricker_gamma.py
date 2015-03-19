import numpy as np
import matplotlib.pyplot as plt
from scipy.special import psi, gamma
from filter import bootstrap_filter, observ_gen

r = np.exp(2)
phi = 1
sigma = np.sqrt(0.3)


def kernel(n, param):
    r = param['r']
    sigma = param['sigma']
    out = np.random.lognormal(mean=param['mean'], sigma=sigma)
    return r*n*np.exp(-n)*out

def kernel_density(n, n_prev, param):
    r = param['r']
    sigma = param['sigma']
    mu = param['mean']
    coeff = r*n_prev*np.exp(-n_prev)
    mean = mu + np.log(coeff)
    out = 1/(n*sigma*np.sqrt(2*np.pi))*np.exp(1/(2*sigma**2)*(np.log(n)-mean)**2)
    return out

def conditional(n, param):
    phi = param['phi']
    out = np.random.poisson(lam=phi*n)
    return out

def proposal(n, y, param):
    r = param['r']
    sigma = param['sigma']
    mu = param['mean']
    alpha, beta = param_gamma(n, r, mu, sigma)
    out = np.random.gamma(shape=y+alpha, scale=beta*(phi*beta+1))
    return out

def initial(param):
    out = np.random.gamma(shape=param['shape'], scale=param['scale'])
    return out

def proposal_density(n, n_prev, y, param):
    sigma = param['sigma']
    mu = param['mean']
    r = param['r']
    alpha, beta = param_gamma(n_prev, r, mu, sigma)
    out = (phi+1/beta)**(alpha+y)/gamma(alpha+y)*n**(alpha+y-1)*np.exp(-(phi+1/beta)*n)
    return  out

def conditional_density(n, y, param):
    phi = param['phi']
    out = np.exp(-phi*n)*(phi*n)**y/np.math.factorial(y)
    return out

def param_gamma(n_prev, r, mu, sigma):
    coeff = r*n_prev*np.exp(-n_prev)
    alpha = 1/sigma**2
    beta = 1/alpha*np.exp(mu+np.log(coeff)+sigma**2/2)
    return alpha, beta

params = {
    'initial': {'shape': 3, 'scale': 1},
    'kernel': {'r': r, 'sigma': sigma, 'mean': 0.0},
    'proposal': {'r': r, 'sigma': sigma, 'mean': 0.0},
    'conditional': {'phi': phi}
    }

observ = observ_gen(100, params, conditional=conditional, initial=initial, kernel=kernel)

observ2 = []
x = 7
for i in range(100):
    observ2.append(x)
    x = kernel(x, params['kernel'])

estim, likeli, ESS = bootstrap_filter(param=params, start=0, end=100, N=1000, kernel_density=kernel_density, conditional_density=conditional_density,
                       proposal=proposal, proposal_density=proposal_density, initial=initial,
                       observations=observ)

mean_esti = [np.mean(est) for est in estim]
plt.plot([i for i in range(100)], mean_esti)
plt.plot([i for i in range(100)], observ2)
plt.savefig('diagno_gamma.pdf')
plt.plot([i for i in range(100)], ESS)
plt.savefig('ESS_gamma.pdf')
plt.plot([i for i in range(101)], [np.log(lik) for lik in likeli])
plt.savefig('loglik.pdf')