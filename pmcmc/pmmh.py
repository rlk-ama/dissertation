import numpy as np
from proposals.proposals import RandomWalkProposal

class PMMH(object):

    def __init__(self, filter, map, iterations, proposals, prior, init, initial, start, end, Ns, adaptation=1000,
                 burnin=1000, target=0.15, target_low=0.1, observations=None, support=None):
        self.filter = filter
        self.map = map
        self.iterations = iterations
        self.proposals = proposals
        self.prior = prior
        self.init = init
        self.start = start
        self.end = end
        self.Ns = Ns
        self.observations = observations
        self.initial = initial
        self.adaptation = adaptation
        self.burnin = burnin
        self.split = 100
        self.steps = self.adaptation//self.split
        self.target = target
        self.target_low = target_low
        if support:
            self.support = support
        else:
            self.support = lambda x: True


    def initalize(self):
        theta = self.init
        return self.routine(theta)

    def routine(self, parameters):
        Map = self.map(*parameters, initial=self.initial, observations=self.observations)
        filter = self.filter(self.start, self.end, self.Ns, Map, proposal={'optimal': True})
        _, _, likeli, _ = next(filter.filter())
        return likeli[-1]


    def accept_reject(self, ratio):
        uniform_draw = np.random.uniform(low=0, high=1, size=None)
        return np.log(uniform_draw) < ratio

    def sub_sample(self, theta, likeli):
            theta_star = [self.proposals[i].sample(theta[i]) for i in range(len(theta))]
            if self.support(theta_star):
                likeli_star = self.routine(theta_star)
                zipped = list(zip(theta, theta_star))
                numerator =  likeli_star + np.prod([self.proposals[i].density(zipped[i][0], zipped[i][1]) for i in range(len(zipped))]) + self.prior.density(theta_star)
                denominator = likeli + np.prod([self.proposals[i].density(zipped[i][1], zipped[i][0]) for i in range(len(zipped))])+ self.prior.density(theta)
                if self.accept_reject(numerator-denominator):
                    theta = theta_star
                    likeli = likeli_star
                    accept = 1
                else:
                    accept = 0
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
        acceptance_rate = np.sum(accepts[0:start])/start
        while not self.target_low < acceptance_rate < self.target and start < self.burnin + self.adaptation:
            self.rescale(acceptance_rate)
            end = start + self.split
            for iteration in range(start, end):
                theta, likeli, accept = self.sub_sample(thetas[iteration], likeli)
                thetas.append(theta)
                accepts.append(accept)
                print(iteration)
            acceptance_rate = ((end-self.burnin-self.split)*acceptance_rate + 2*np.sum(accepts[start:end]))/(end-self.burnin)
            start = start + self.split

        for iteration in range(start, self.iterations):
            theta, likeli, accept = self.sub_sample(thetas[iteration], likeli)
            thetas.append(theta)
            accepts.append(accept)
            print(iteration)

        return thetas, accepts

    def rescale(self, acceptance_rate):
        coeff = abs((acceptance_rate - self.target)/self.target)
        new = [self.proposals[i].sigma*(1+coeff) if acceptance_rate > self.target else self.proposals[i].sigma/(1+coeff) for i in range(len(self.proposals))]
        self.proposals = [RandomWalkProposal(sigma=sigma) for sigma in new]
        print(acceptance_rate, new)