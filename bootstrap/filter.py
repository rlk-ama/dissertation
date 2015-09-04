import numpy as np

from distributions.distributions2 import Normal

DTYPE = np.float64

class BootstrapFilter(object):

    def __init__(self, end, Ns, Map, initial=None, proposal=None, likeli=False, **kwargs):
        self.start = 0
        self.end = end
        self.Ns = Ns if isinstance(Ns, list) else [Ns]
        self.multiple = True if isinstance(Ns, list) else False
        self.Map = Map
        self.proposal = proposal if proposal else {'prior': False, 'optimal': True}
        self.observations = self.Map.observations
        self.likeli = likeli

        if initial:
            self.initial = initial
        else:
            self.initial = {'distribution': Normal, 'loc': 0, 'scale': 1}

    #@profile
    def sub_filter(self, N, prior, lik=False):
        proposal = 'prior' if prior else 'optimal'
        if not lik:
            particles_all = np.zeros(shape=(self.end-self.start, N))
            ESS = np.zeros(self.end-self.start)
        likelihoods = np.zeros(self.end-self.start)

        likelihood = 0
        particles = self.initial['distribution']().sample(**{k:np.array([v]*N, dtype=np.float64) for k,v in self.initial.items() if k != 'distribution'}) #draw from initial distribution
        weights = np.array([1.0/N]*N, dtype=np.float64)

        likelihoods[0] = likelihood

        for i in range(self.start+1, self.end):
            lweights = np.log(weights)
            weights = np.exp(lweights - max(lweights))
            np.divide(weights, sum(weights), out=weights)
            if not lik:
                ESS[i-1] = (1/sum(np.multiply(weights, weights)))

            indices = np.random.multinomial(N, weights, size=1)[0]
            ancestors = np.array([particles[j] for j in range(len(indices)) for _ in range(indices[j])], dtype=DTYPE)

            if prior:
                particles = self.Map.prior.sample(ancestors)
            else:
                particles = self.Map.proposal.sample(ancestors, self.observations[i])

            obs = self.Map.conditional.density(particles, self.observations[i])

            if prior:
                weights = obs.base
            else:
                denom = self.Map.proposal.density(particles, ancestors, self.observations[i])
                transi = self.Map.kernel.density(particles, ancestors)
                np.multiply(obs, transi, out=weights)
                np.divide(weights, denom, out=weights)

            if sum(weights) == 0:
                likelihoods[i:].fill(-np.inf)
                if not lik:
                    particles_all[i-1] = ancestors
                    return proposal, particles_all, likelihoods, ESS
                else:
                    return proposal, likelihoods

            likelihood += -np.log(N) + np.log(sum(weights))

            likelihoods[i] = likelihood
            if not lik:
                particles_all[i-1] = ancestors

        if not lik:
            np.divide(weights, sum(weights), out=weights)
            ESS[self.end-1] = 1/sum(np.multiply(weights, weights))
            indices = np.random.multinomial(N, weights, size=1)[0]
            ancestors = np.array([particles[j] for j in range(len(indices)) for _ in range(indices[j])], dtype=DTYPE)
            particles_all[self.end-1] = ancestors
            return proposal, particles_all, likelihoods, ESS
        else:
            return proposal, likelihoods

    def filter(self):
        for N in self.Ns:
            if self.proposal.get('prior', False):
                yield self.sub_filter(N, True, self.likeli)
            if self.proposal.get('optimal', False):
                yield self.sub_filter(N, False, self.likeli)