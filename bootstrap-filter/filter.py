import numpy


numpy.seterr(over="raise", under="raise", invalid="raise")

class BootstrapFilter(object):

    def __init__(self, start, end, N, Map):
        self.start = start
        self.end = end
        self.N = N
        self.Map = Map
        self.observations, self.state = self.observ_gen()

    def observ_gen(self):
        observ = []
        state = []
        x = self.Map.initial()
        for i in range(self.end+1):
            observ.append(self.Map.conditional(x))
            state.append(x)
            x = self.Map.kernel(x)
        return observ, state

    def filter(self):
        particles_all = numpy.empty(shape=(self.end-self.start, self.N))
        particles = []

        weights = numpy.empty(shape=self.N, dtype='float64')
        likelihoods = []
        ESS = []
        likelihood = 0

        for i in range(self.N):
            particles.append(self.Map.initial()) #draw from initial distribution
            weights[i] = 1.0/self.N

        for i in range(self.start+1, self.end+1):
            numpy.divide(weights, sum(weights), out=weights)

            try: #catch underflow error
                ESS.append(1/sum([weight**2 for weight in weights]))
            except FloatingPointError:
                ESS.append(0)

            indices = numpy.random.multinomial(self.N, weights, size=1)[0]
            ancestors = [particles[j] for j in range(len(indices)) for _ in range(indices[j])]
            particles = [self.Map.proposal(ancestor, self.observations[i]) for ancestor in ancestors]

            denom = [self.Map.proposal_density(particles[ancestors.index(ancestor)], ancestor, self.observations[i]) for ancestor in ancestors] #q(x_t|x_t-1, y_t)
            obs = [self.Map.conditional_density(particle, self.observations[i]) for particle in particles] #p(y_t[x_t)
            transi = [self.Map.kernel_density(particles[ancestors.index(ancestor)], ancestor) for ancestor in ancestors] #p(x_t|x_t-1)

            weights = numpy.zeros(self.N)
            numpy.multiply(obs, transi, out=weights)
            numpy.divide(weights, denom, out=weights)
            likelihood += -numpy.log(self.N) + numpy.log(sum(weights))
            likelihoods.append(likelihood)
            particles_all[i-1] = ancestors

        return particles_all, likelihoods, ESS