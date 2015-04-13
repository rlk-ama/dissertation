import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import broyden1, fsolve
from scipy.special import psi, gamma, polygamma, gammaln
from filter import bootstrap_filter, observ_gen

r = np.exp(3.5)
phi = 10
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

def kernel_prior(n, obs, param):
    r = param['r']
    sigma = param['sigma']
    out = np.random.lognormal(mean=param['mean'], sigma=sigma)
    return r*n*np.exp(-n)*out

def kernel_density_prior(n, n_prev, obs, param):
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
    out = np.random.gamma(shape=y+alpha, scale=beta/(phi*beta+1))
    return out

def initial(param):
    out = np.random.gamma(shape=param['shape'], scale=param['scale'])
    return out

def proposal_density(n, n_prev, y, param):
    sigma = param['sigma']
    mu = param['mean']
    r = param['r']
    alpha, beta = param_gamma(n_prev, r, mu, sigma)
    outlog= (alpha+y)*np.log((phi*beta+1)/beta) - gammaln(alpha+y) + (alpha+y-1)*np.log(n) - ((phi*beta+1)/beta)*n
    #out = ((phi*beta+1)/beta)**(alpha+y)/gamma(alpha+y)*n**(alpha+y-1)*np.exp(-((phi*beta+1)/beta)*n)
    out = np.exp(outlog)
    return  out

def conditional_density(n, y, param):
    phi = param['phi']
    outlog = -phi*n + y*np.log(phi*n) - gammaln(y+1)
    #out = np.exp(-phi*n)*(phi*n)**y/np.math.factorial(y)
    out = np.exp(outlog)
    return out

def param_gamma(n_prev, r, mu, sigma):
    coeff = r*n_prev*np.exp(-n_prev)
    alpha = 1/sigma**2
    beta = 1/alpha*np.exp(mu+np.log(coeff)+sigma**2/2)
    return alpha, beta

def param_gamma2(n_prev, r, mu, sigma):

    def p(k):
        return -k**3 + ((sigma**2-1)/(3*sigma**2))*k**2 + k/sigma**2 - ((15+2*sigma**2)/(30*sigma**2))

    coeff = r*n_prev*np.exp(-n_prev)
    alpha = fsolve(p, 0.01)[0]
    beta = 1/alpha*np.exp(mu+np.log(coeff)+ 1/(2*alpha))
    return alpha, beta

if __name__ == '__main__':
     #liks = []
     # liks2 = []
     # rs = map(np.exp, np.linspace(2.5,4.5, 190))
     # for r in rs:

    params = {
    'initial': {'shape': 3, 'scale': 1},
    'kernel': {'r': r, 'sigma': sigma, 'mean': 0.0},
    'proposal': {'r': r, 'sigma': sigma, 'mean': 0.0},
    'conditional': {'phi': phi}
    }

    observ, state = observ_gen(30, params, conditional=conditional, initial=initial, kernel=kernel)

    estim, likeli, ESS = bootstrap_filter(param=params, start=0, end=30, N=1000, kernel_density=kernel_density, conditional_density=conditional_density,
                            proposal=proposal, proposal_density=proposal_density, initial=initial,
                            observations=observ)

        # estim2, likeli2, ESS2 = bootstrap_filter(param=params, start=0, end=30, N=100, kernel_density=kernel_density, conditional_density=conditional_density,
        #                        proposal=kernel_prior, proposal_density=kernel_density_prior, initial=initial,
        #                        observations=observ)
        # liks.append(likeli[-1])
        #liks2.append(likeli2[-1])

    fig, ax1 = plt.subplots()
    #fig1 = plt.figure()
    #mean_esti = [np.mean(est) for est in estim]
    # ax1.plot([i for i in range(30)], mean_esti)
    #plt.plot([i for i in range(30)], mean_esti)
    ax1.plot([i for i in range(31)], state)
    # plt.savefig('diagno_gamma.pdf')
    # plt.show()
    # plt.close()
    ax2 = ax1.twinx()
    ax2.plot([i for i in range(30)], ESS, color="red")
    # plt.savefig('ESS_gamma.pdf')
    # plt.show()
    # plt.close()
    #fig3 = plt.figure()
    #plt.plot(rs, liks)
    #plt.plot(rs, liks2)
    #plt.show()
    #plt.plot([i for i in range(30)], likeli)
    # plt.savefig('loglik_gamma.pdf')
    # plt.close()
    #
    #mean_esti = [np.mean(est) for est in estim2]
    # fig1 = plt.figure()
    #plt.plot([i for i in range(30)], mean_esti)
    #plt.show()
    #plt.plot([i for i in range(31)], state)
    # plt.savefig('diagno_prior.pdf')
    # plt.close()
    #fig2 = plt.figure()
    #plot([i for i in range(30)], ESS)
    # plt.savefig('ESS_prior.pdf')
    # plt.close()
    # fig3 = plt.figure()
    #plt.plot([i for i in range(30)], likeli2)
    # plt.savefig('loglik_prior.pdf')
    # plt.close()
    plt.show()
