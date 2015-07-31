import numpy as np


DTYPE = np.float64

class BootstrapFilter(object):

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
            indices = np.random.multinomial(N, weights, size=1)[0]

            particles = self.Map.proposal.sample(indices, self.observations)
            denom = self.Map.proposal.density(particles, indices, self.observations)
            kernel = self.Map.kernel.density(particles, indices, self.observations)

            num = 0
            for i in range(self.rep):
                cond = self.Map.conditional.density(indices, particles, self.observations)
                num += cond
            num = num/self.rep

            np.multiply(num, kernel, out=weights)
            np.divide(weights, denom, out=weights)

            if sum(weights) == 0:
                likelihoods[i:].fill(-np.inf)

                if not self.likeli:
                    return particles_all, likelihoods, ESS
                else:
                    return  likelihoods

            likelihood += -np.log(N) + np.log(sum(weights))

            likelihoods[i] = likelihood
            self.rotate(indices, self.observations)

            if not self.likeli:
                particles_all[i-1] = particles

        if not self.likeli:
            weights = self.normalize(weights)
            ESS[self.end-1] = 1/sum(np.multiply(weights, weights))
            indices = np.random.multinomial(N, weights, size=1)[0]
            ancestors = np.array([self.observations[-1, j] for j in range(len(indices)) for _ in range(indices[j])], dtype=DTYPE)
            particles_all[self.end-1] = ancestors
            return particles_all, likelihoods, ESS
        else:
            return likelihoods

    def rotate(self, indices, observations):
        new = np.array([observations[-1, j] for j in range(len(indices)) for _ in range(indices[j])], dtype=DTYPE)
        self.observations = np.roll(self.observations, -1, 0)
        self.observations[-1] = 0

    def normalize(self, weights):
        lweights = np.log(weights)
        weights = np.exp(lweights - max(lweights))
        np.divide(weights, sum(weights), out=weights)
        return weights

    def filter(self):
        for N in self.Ns:
            yield self.sub_filter(N, True)