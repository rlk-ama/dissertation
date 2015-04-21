import numpy as np
from scipy.optimize import broyden1, fsolve
from scipy.special import psi, gamma, polygamma, gammaln

class RickerMap(object):

    def __init__(self, r, phi, mu, sigma, initial, approx='simple'):
        self.r = r
        self.phi = phi
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.approx = approx

    def kernel(self, n):
        out = np.random.lognormal(mean=self.mu, sigma=self.sigma)
        return self.r*n*np.exp(-n)*out

    def kernel_density(self, n, n_prev):
        coeff = self.r*n_prev*np.exp(-n_prev)
        mean = self.mu + np.log(coeff)
        out = 1/(n*self.sigma*np.sqrt(2*np.pi))*np.exp(-1/(2*self.sigma**2)*(np.log(n)-mean)**2)
        return out

    def prior_proposal(self, n, obs):
        out = np.random.lognormal(mean=self.mu, sigma=self.sigma)
        return self.r*n*np.exp(-n)*out

    def prior_proposal_density(self, n, n_prev, obs):
        coeff = self.r*n_prev*np.exp(-n_prev)
        mean = self.mu + np.log(coeff)
        out = 1/(n*self.sigma*np.sqrt(2*np.pi))*np.exp(-1/(2*self.sigma**2)*(np.log(n)-mean)**2)
        return out

    def conditional(self, n):
        out = np.random.poisson(lam=self.phi*n)
        return out

    def proposal(self, n, y):
        if self.approx == 'simple':
            alpha, beta = self.param_gamma(n)
        else:
            alpha, beta = self.param_gamma2(n)
        out = np.random.gamma(shape=y+alpha, scale=beta/(self.phi*beta+1))
        return out

    def proposal_density(self, n, n_prev, y):
        if self.approx == 'simple':
            alpha, beta = self.param_gamma(n_prev)
        else:
            alpha, beta = self.param_gamma2(n_prev)
        outlog= (alpha+y)*np.log((self.phi*beta+1)/beta) - gammaln(alpha+y) + (alpha+y-1)*np.log(n) - ((self.phi*beta+1)/beta)*n
        #out = ((phi*beta+1)/beta)**(alpha+y)/gamma(alpha+y)*n**(alpha+y-1)*np.exp(-((phi*beta+1)/beta)*n)
        out = np.exp(outlog)
        return  out

    def conditional_density(self, n, y):
        outlog = -self.phi*n + y*np.log(self.phi*n) - gammaln(y+1)
        #out = np.exp(-phi*n)*(phi*n)**y/np.math.factorial(y)
        out = np.exp(outlog)
        return out

    def param_gamma(self, n_prev):
        coeff = self.r*n_prev*np.exp(-n_prev)
        alpha = 1/self.sigma**2
        beta = 1/alpha*np.exp(self.mu+np.log(coeff)+self.sigma**2/2)
        return alpha, beta

    def param_gamma2(self, n_prev):

        coeff = self.r*n_prev*np.exp(-n_prev)
        alpha = (6-self.sigma**2)/(3*self.sigma**2)
        beta = 1/alpha*np.exp(self.mu+np.log(coeff)+ 1/(2*alpha))
        return alpha, beta