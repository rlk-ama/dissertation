import numpy as np

class PMMH(object):

    def __init__(self, filter, map, iterations, proposal, prior, init, initial, start, end, Ns, observations=None):
        self.filter = filter
        self.map = map
        self.iterations = iterations
        self.proposal = proposal
        self.prior = prior
        self.init = init
        self.start = start
        self.end = end
        self.Ns = Ns
        self.observations = observations
        self.initial = initial


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

    def sample(self):
        likeli = self.initalize()
        theta = self.init
        thetas = [theta]
        steps = []
        for iteration in range(self.iterations):
            theta_star = [self.proposal.sample(param) for param in thetas[iteration]]
            likeli_star = self.routine(theta_star)
            numerator =  likeli_star + np.prod([self.proposal.density(t, t_star) for t, t_star in zip(theta, theta_star)]) + self.prior.density(theta_star)
            denominator = likeli + np.prod([self.proposal.density(t_star, t) for t, t_star in zip(theta, theta_star)])+ self.prior.density(theta)
            if self.accept_reject(numerator-denominator):
                theta = theta_star
                likeli = likeli_star
                steps.append(1)
            else:
                steps.append(0)
            thetas.append(theta)

        return thetas, steps
