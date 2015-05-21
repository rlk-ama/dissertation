import numpy
import sys

numpy.seterr(over="raise", under="raise", invalid="raise")

class BootstrapFilter(object):

    def __init__(self, start, end, Ns, Map, proposal={'prior': False, 'optimal': True}):
        self.start = start
        self.end = end
        self.Ns = Ns if isinstance(Ns, list) else [Ns]
        self.multiple = True if isinstance(Ns, list) else False
        self.Map = Map
        self.proposal = proposal
        self.observations = self.Map.observations

    def sub_filter(self, N, prior):
        particles_all = numpy.zeros(shape=(self.end-self.start, N))
        particles = []

        weights = numpy.zeros(shape=N, dtype='float64')
        likelihoods = []
        ESS = []
        likelihood = 0

        for i in range(N):
            particles.append(self.Map.initial.sample()) #draw from initial distribution
            weights[i] = 1.0/N

        for i in range(self.start+1, self.end+1):
            numpy.divide(weights, sum(weights), out=weights)
            try: #catch underflow error
                ESS.append(1/sum([weight**2 for weight in weights]))
            except FloatingPointError:
                ESS.append(0)

            indices = numpy.random.multinomial(N, weights, size=1)[0]
            ancestors = [particles[j] for j in range(len(indices)) for _ in range(indices[j])]
            if prior:
                particles = [self.Map.prior.sample(ancestor) for ancestor in ancestors]
            else:
                particles = [self.Map.proposal.sample(ancestor, self.observations[i]) for ancestor in ancestors]

            if prior:
                denom = [self.Map.prior.density(particles[j], ancestors[j]) for j in range(len(ancestors))] #q(x_t|x_t-1, y_t)
            else:
                denom = [self.Map.proposal.density(particles[j], ancestors[j], self.observations[i]) for j in range(len(ancestors))]
            obs = [self.Map.conditional.density(particle, self.observations[i]) for particle in particles] #p(y_t[x_t)
            obs = [ob if ob != sys.float_info[3] else 0 for ob in obs ]
            transi = [self.Map.kernel.density(particles[j], ancestors[j]) for j in range(len(ancestors))] #p(x_t|x_t-1)

            weights = numpy.zeros(N)
            numpy.multiply(obs, transi, out=weights)
            numpy.divide(weights, denom, out=weights)
            likelihood += -numpy.log(N) + numpy.log(sum(weights))
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