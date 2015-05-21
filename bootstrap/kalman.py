import numpy as np

class KalmanMap(object):

    def __init__(self, phi, sigma_state, sigma_obs, initial):
        self.phi = phi
        self.sigma_state = sigma_state
        self.sigma_obs = sigma_obs
        self.initial = initial

    @staticmethod
    def normal_density(x, mean, sd):
        return 1/(np.sqrt(2*np.pi)*sd)*np.exp(-1/(2*sd**2)*(x - mean)**2)

    def kernel_density(self, x, x_prev):
        mean =self.phi*x_prev
        out = KalmanMap.normal_density(x, mean, self.sigma_state)
        return out

    def kernel(self, x):
        mean = self.phi*x
        return np.random.normal(mean, self.sigma_state)


    def conditional(self, x):
        mean = x
        out = np.random.normal(mean, self.sigma_obs)
        return out

    def conditional_density(self, x, y):
        mean = x
        out = KalmanMap.normal_density(y, mean, self.sigma_obs)
        return out

    def prior_proposal(self, x, y):
        return self.kernel(x)

    def prior_proposal_density(self, x, x_prev, y):
        return self.kernel_density(x, x_prev)

    def proposal(self, x, y):
        coeff = (self.sigma_obs*self.sigma_state)**2/(self.sigma_obs**2+self.sigma_state**2)
        mean = coeff*(self.phi*x/self.sigma_state**2 + y/self.sigma_obs**2)
        sd = np.sqrt(coeff)
        return np.random.normal(mean, sd)

    def proposal_density(self, x, x_prev, y):
        coeff = (self.sigma_obs*self.sigma_state)**2/(self.sigma_obs**2+self.sigma_state**2)
        mean = coeff*(self.phi*x_prev/self.sigma_state**2 + y/self.sigma_obs**2)
        sd = np.sqrt(coeff)
        return KalmanMap.normal_density(x, mean, sd)