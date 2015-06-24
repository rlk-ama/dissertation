import numpy as np

from distributions.distributions2 import Normal

DTYPE = np.float64

class BootstrapFilter(object):

    def __init__(self, start, end, Ns, Map, initial=None, proposal=None):
        self.start = start
        self.end = end
        self.Ns = Ns if isinstance(Ns, list) else [Ns]
        self.multiple = True if isinstance(Ns, list) else False
        self.Map = Map
        self.proposal = proposal if proposal else {'prior': False, 'optimal': True}
        self.observations = self.Map.observations

        if initial:
            self.initial = initial
        else:
            self.initial = {'distribution': Normal, 'loc': 0, 'scale': 1}

    #@profile
    def sub_filter(self, N, prior):
        particles_all = np.zeros(shape=(self.end-self.start, N))
        particles = np.empty(N)

        likelihoods = []
        ESS = []
        likelihood = 0
        particles = self.initial['distribution']().sample(**{k:np.array([v]*N, dtype=np.float64) for k,v in self.initial.items() if k != 'distribution'}) #draw from initial distribution
        weights = np.array([1.0/N]*N, dtype=np.float64)

        for i in range(self.start+1, self.end+1):
            np.divide(weights, sum(weights), out=weights)
            ESS.append(1/sum(np.multiply(weights, weights)))

            indices = np.random.multinomial(N, weights, size=1)[0]
            ancestors = np.array([particles[j] for j in range(len(indices)) for _ in range(indices[j])], dtype=DTYPE)

            if prior:
                particles = self.Map.prior.sample(ancestors)
            else:
                particles = self.Map.proposal.sample(ancestors, self.observations[i])

            if prior:
                denom = self.Map.prior.density(particles, ancestors) #q(x_t|x_t-1, y_t)
            else:
                denom = self.Map.proposal.density(particles, ancestors, self.observations[i])

            obs = self.Map.conditional.density(particles, self.observations[i])
            transi = self.Map.kernel.density(particles, ancestors)
            weights_n = np.zeros(N)
            np.multiply(obs, transi, out=weights_n)
            np.divide(weights_n, denom, out=weights_n)
            if sum(weights_n) == 0:
                continue
            weights = np.copy(weights_n)
            likelihood += -np.log(N) + np.log(sum(weights))
            likelihoods.append(likelihood)
            particles_all[i-1] = ancestors

        return particles_all, likelihoods, ESS


    def filter(self):
        for N in self.Ns:
            if self.proposal.get('prior', False):
                particles_all, likelihoods, ESS = self.sub_filter(N, True)
                yield 'prior', particles_all, likelihoods, ESS
            if self.proposal.get('optimal', False):
                particles_all, likelihoods, ESS = self.sub_filter(N, False)
                yield 'optimal', particles_all, likelihoods, ESS