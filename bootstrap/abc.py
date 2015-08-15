import numpy as np


DTYPE = np.float64

class ABCFilter(object):

    def __init__(self, end, Ns, Map, rep=100, likeli=False, proposal=None, initial=None, adaptation=False):
        self.end = end
        self.Ns = Ns if isinstance(Ns, list) else [Ns]
        self.multiple = True if isinstance(Ns, list) else False
        self.Map = Map
        self.observations = self.Map.observations
        self.likeli = likeli
        self.rep = rep
        self.proposal = proposal
        self.start = self.Map.tau
        self.adaptation = adaptation

    #@profile
    def sub_filter(self, N, proposal, likeli=False):

        if not self.likeli:
            particles_all = []

        likelihoods = np.zeros(self.end-self.start-1)
        ESS = np.zeros(self.end-self.start)
        likelihood = 0
        weights = np.array([1.0/N]*N, dtype=np.float64)
        first_val = self.observations[self.start]

        for i in range(self.start+1, self.end):
            weights = self.normalize(weights)

            ESS[i-self.start-1] = (1/sum(np.multiply(weights, weights)))

            particles = self.Map.proposal.sample(self.observations, i, N)
            denom = self.Map.proposal.density(particles, self.observations, i)
            if proposal == "prior":
                kernel = denom
            else:
                kernel = self.Map.kernel.density(particles, self.observations, i)

            if self.adaptation:
                m = int(self.rep*self.observations[i]/first_val)
            else:
                m = self.rep
            num = [0]*len(particles)
            for j in range(m):
                cond = self.Map.conditional.density(particles, self.observations, i)
                num = num + cond
            num = np.divide(num, self.rep)

            np.multiply(num, kernel, out=weights)
            np.divide(weights, denom, out=weights)

            if sum(weights) == 0:
                likelihoods[i-self.start-1:].fill(-np.inf)

                if not self.likeli:
                    return particles_all, likelihoods, ESS
                else:
                    return likelihoods

            likelihood += -np.log(N) + np.log(sum(weights))

            likelihoods[i-self.start-1] = likelihood

            if not self.likeli:
                particles_all.append(particles)

        if not self.likeli:
            weights = self.normalize(weights)
            ESS[self.end-self.start-1] = 1/sum(np.multiply(weights, weights))
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
            if self.proposal.get("prior", None):
                yield 'prior', self.sub_filter(N, likeli=self.likeli, proposal="prior")
            else:
                yield 'optimal', self.sub_filter(N, likeli=self.likeli, proposal="optimal")