import numpy as np
from time import time
from proposals.proposals import RandomWalkProposal

class PMMH(object):

    def __init__(self, filter, map, iterations, proposals, prior, init, initial, end, Ns, adaptation=1000,
                 burnin=1500, target=0.15, target_low=0.10, observations=None, support=None, initial_filter=None,
                 filter_proposal='optimal', *args, **kwargs):
        self.filter = filter
        self.map = map
        self.iterations = iterations
        self.proposals = proposals
        self.prior = prior
        self.init = init
        self.end = end
        self.Ns = Ns
        self.observations = observations
        self.initial = initial
        self.adaptation = adaptation
        self.burnin = burnin
        self.split = 100
        self.target = target
        self.target_low = target_low
        self.initial_filter = initial_filter
        self.filter_proposal = filter_proposal
        if support:
            self.support = support
        else:
            self.support = lambda x: True
        self.kwargs = kwargs

    def initalize(self):
        theta = self.init
        return self.routine(theta)

    def routine(self, parameters):
        try:
            Map = self.map(*parameters, initial=self.initial, observations=self.observations, **self.kwargs)
        except:
            return -np.inf
        filter = self.filter(self.end, self.Ns, Map, proposal={self.filter_proposal: True}, initial=self.initial_filter,
                             likeli=True)
        _, likeli= next(filter.filter())
        return likeli[-1]


    def accept_reject(self, ratio):
        uniform_draw = np.random.uniform(low=0, high=1, size=None)
        return np.log(uniform_draw) < ratio

    def sub_sample(self, theta, likeli, index=None):
        if index:
            theta_star = theta
            theta_star[index] = RandomWalkProposal(sigma=self.proposals.lambdas[index]*self.proposals.cov[index, index]).sample(theta[index]).base
        else:
            theta_star = self.proposals.sample(theta)
        if self.support(theta_star):
            likeli_star = self.routine(theta_star)
            numerator =  likeli_star + np.log(self.proposals.density(theta, theta_star)) + np.log(self.prior.density(np.array(theta_star)))
            denominator = likeli + np.log(self.proposals.density(theta_star, theta)) + np.log(self.prior.density(np.array(theta)))
            if self.accept_reject(numerator-denominator):
                theta = theta_star
                likeli = likeli_star
            accept = min(1, np.exp(numerator-denominator))
        else:
            accept = 0
        return theta, likeli, accept

    def sample(self):
        likeli = self.initalize()
        theta = self.init
        thetas = [theta]
        accepts = []
        for iteration in range(self.burnin):
            theta, likeli, accept = self.sub_sample(thetas[iteration], likeli)
            thetas.append(theta)
            accepts.append(accept)
            print(iteration)

        start = self.burnin
        k = 2
        while start < self.burnin + self.adaptation:
            end = start + self.split
            index = list(np.random.multinomial(1, [1/len(theta)]*len(theta))).index(1)
            mean = []
            for iteration in range(start, end):
                theta, likeli, accept = self.sub_sample(thetas[iteration], likeli, index)
                thetas.append(theta)
                accepts.append(accept)
                mean.append(theta)
                print(iteration)
            acceptance_rate = np.sum(accepts[start:end])/self.split
            print(acceptance_rate)
            self.rescale(acceptance_rate, k, index, mean)
            start = start + self.split
            k += 1

        for iteration in range(start, self.iterations):
            theta, likeli, accept = self.sub_sample(thetas[iteration], likeli)
            thetas.append(theta)
            accepts.append(accept)
            print(iteration)

        return thetas, accepts

    def rescale(self, acceptance_rate, k, index, mean):
        coeff = (acceptance_rate - self.target)/k
        self.proposals.lambdas[index] = self.proposals.lambdas[index]*np.exp(coeff)
        self.proposals.mean = self.proposals.mean + 1/k*(np.mean(mean, axis=0)-self.proposals.mean)
        x = np.mean(mean, axis=0)-self.proposals.mean
        xT = x[:, np.newaxis]
        self.proposals.cov = self.proposals.cov + 1/k*(x*xT)