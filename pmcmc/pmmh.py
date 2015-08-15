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
        if 'tol' in kwargs:
            self.tol = kwargs['tol']

    def initalize(self):
        theta = self.init
        return self.routine(theta)

    def routine(self, parameters):
        try:
            Map = self.map(*parameters, initial=self.initial, observations=self.observations, tol=self.tol)
        except:
            return -np.inf
        filter = self.filter(self.end, self.Ns, Map, proposal={self.filter_proposal: True}, initial=self.initial_filter,
                             likeli=True)
        _, likeli= next(filter.filter())
        return likeli[-1]


    def accept_reject(self, ratio):
        uniform_draw = np.random.uniform(low=0, high=1, size=None)
        return np.log(uniform_draw) < ratio

    def sub_sample(self, theta, likeli):
            theta_star = [self.proposals[i].sample(theta[i])[0] for i in range(len(theta))]
            if self.support(theta_star):
                likeli_star = self.routine(theta_star)
                zipped = list(zip(theta, theta_star))
                numerator =  likeli_star + np.sum([np.log(self.proposals[i].density(zipped[i][0], zipped[i][1])) for i in range(len(zipped))]) + np.log(self.prior.density(np.array(theta_star)))
                denominator = likeli + np.sum([np.log(self.proposals[i].density(zipped[i][1], zipped[i][0])) for i in range(len(zipped))]) + np.log(self.prior.density(np.array(theta)))
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
        acceptance_rate = np.sum(accepts[start-self.split:start])/self.split
        print(acceptance_rate)
        k = 1
        while start < self.burnin + self.adaptation:
            self.rescale(acceptance_rate, k)
            end = start + self.split
            for iteration in range(start, end):
                theta, likeli, accept = self.sub_sample(thetas[iteration], likeli)
                thetas.append(theta)
                accepts.append(accept)
                print(iteration)
            acceptance_rate = np.sum(accepts[start:end])/self.split
            print(acceptance_rate)
            start = start + self.split
            k += 1

        for iteration in range(start, self.iterations):
            theta, likeli, accept = self.sub_sample(thetas[iteration], likeli)
            thetas.append(theta)
            accepts.append(accept)
            print(iteration)

        return thetas, accepts

    def rescale(self, acceptance_rate, k):
        coeff = (acceptance_rate - self.target)/k
        news = [self.proposals[i].sigma*np.exp(coeff) for i in range(len(self.proposals))]
        self.proposals = [RandomWalkProposal(sigma=new) for new in news]
        print(news)