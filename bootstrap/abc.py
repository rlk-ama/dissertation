import numpy as np


DTYPE = np.float64

class ABCFilter(object):

    def __init__(self, start, end, Ns, Map, rep, likeli=False):
        self.start = start
        self.end = end
        self.Ns = Ns if isinstance(Ns, list) else [Ns]
        self.multiple = True if isinstance(Ns, list) else False
        self.Map = Map
        self.observations = self.Map.observations
        self.likeli = likeli
        self.rep = rep

    #@profile
    def sub_filter(self, N, prior, likeli=False):

        if not self.likeli:
            particles_all = []

        likelihoods = np.zeros(self.end-self.start)
        ESS = np.zeros(self.end-self.start)
        likelihood = 0
        weights = np.array([1.0/N]*N, dtype=np.float64)

        likelihoods[0] = likelihood

        for i in range(self.start+1, self.end):
            weights = self.normalize(weights)

            ESS[i-1] = (1/sum(np.multiply(weights, weights)))

            particles = self.Map.proposal.sample(self.observations, i, N)
            denom = self.Map.proposal.density(particles, self.observations, i)
            kernel = self.Map.kernel.density(particles, self.observations, i)

            num = [0]*len(particles)
            for i in range(self.rep):
                cond = self.Map.conditional.density(particles, self.observations, i)
                num = num + cond
            num = np.divide(num, self.rep)

            np.multiply(num, kernel, out=weights)
            np.divide(weights, denom, out=weights)

            if sum(weights) == 0:
                likelihoods[i:].fill(-np.inf)

                if not self.likeli:
                    return particles_all, likelihoods, ESS
                else:
                    return likelihoods

            likelihood += -np.log(N) + np.log(sum(weights))

            likelihoods[i] = likelihood

            if not self.likeli:
                particles_all[i-1] = particles

        if not self.likeli:
            weights = self.normalize(weights)
            ESS[self.end-1] = 1/sum(np.multiply(weights, weights))
            return particles_all, likelihoods, ESS
        else:
            return likelihoods


    def normalize(self, weights):
        lweights = np.log(weights)
        weights = np.exp(lweights - max(lweights))
        np.divide(weights, sum(weights), out=weights)
        return weights

    def filter(self):
        for N in self.Ns:
            yield self.sub_filter(N, self.likeli)