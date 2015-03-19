import numpy


numpy.seterr(over="raise", under="raise", invalid="raise")

def bootstrap_filter(param, start, end, N, kernel_density, conditional_density, proposal, proposal_density, initial, observations):
    particles_all = numpy.empty(shape=(end-start, N))
    particles = []

    weights = numpy.empty(shape=N, dtype='float64')
    likelihoods = []
    ESS = []

    for i in range(N):
        particles.append(initial(param['initial'])) #draw from initial distribution
        weights[i] = 1.0/N
    likelihood = -numpy.log(N) + numpy.log(sum(weights))
    likelihoods.append(likelihood)

    for i in range(start+1, end+1):
        numpy.divide(weights, sum(weights), out=weights)

        try: #catch underflow error
            ESS.append(1/sum([weight**2 for weight in weights]))
        except FloatingPointError:
            ESS.append(0)

        indices = numpy.random.multinomial(N, weights, size=1)[0]
        ancestors = [particles[j] for j in range(len(indices)) for _ in range(indices[j])]
        particles = [proposal(ancestor, observations[i], param['proposal']) for ancestor in ancestors]

        denom = [proposal_density(particles[ancestors.index(ancestor)],
                                  ancestor, observations[i], param['proposal']) for ancestor in ancestors] #q(x_t|x_t-1, y_t)
        obs = [conditional_density(particle, observations[i],
                                   param['conditional']) for particle in particles] #p(y_t[x_t)
        transi = [kernel_density(particles[ancestors.index(ancestor)],
                                 ancestor, param['kernel']) for ancestor in ancestors] #p(x_t|x_t-1)

        weights = numpy.zeros(N)
        numpy.multiply(obs, transi, out=weights)
        numpy.divide(weights, denom, out=weights)
        likelihood += -numpy.log(N) + numpy.log(sum(weights))
        likelihoods.append(likelihood)
        particles_all[i-1] = ancestors

    return particles_all, likelihoods, ESS

def observ_gen(N, params, conditional, initial, kernel):
    observ = []
    state = []
    x = initial(params['initial'])
    for i in range(N+1):
        observ.append(conditional(x, params['conditional']))
        state.append(x)
        x = kernel(x, params['kernel'])
    return observ, state