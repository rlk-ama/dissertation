import numpy

def _normalize(l):
    total = sum(l)
    return [item/total for item in l]

def bootstrap_filter(param, start, end, N, kernel_density, conditional_density, proposal, proposal_density, initial, observations):
    particles_all = numpy.empty(shape=(end-start, N))
    particles = []
    weights = []
    likelihoods = []
    ESS = []

    for i in range(N):
        particles.append(initial(param['initial']))
        weights.append(1.0/N)
    likelihood = 1.0/N*sum(weights)
    likelihoods.append(likelihood)

    for i in range(start+1, end+1):
        weights_norm = _normalize(weights)
        indices = numpy.random.multinomial(N, weights_norm, size=1)[0]
        ancestors = [particles[j] for j in range(len(indices)) for _ in range(indices[j])]
        particles = [proposal(ancestor, observations[i], param['proposal']) for ancestor in ancestors]
        denom = [proposal_density(particles[ancestors.index(ancestor)], ancestor, observations[i], param['proposal']) for ancestor in ancestors]
        obs = [conditional_density(particle, observations[i], param['conditional']) for particle in particles]
        transi = [kernel_density(particles[ancestors.index(ancestor)], ancestor, param['kernel']) for ancestor in ancestors]
        weights = numpy.zeros(N)
        numpy.multiply(obs, transi, out=weights)
        numpy.divide(weights, denom, out=weights)
        likelihood *= 1.0/N*sum(weights)
        likelihoods.append(likelihood)
        ESS.append(1/sum([weight**2 for weight in weights_norm]))
        particles_all[i-1] = ancestors

    return particles_all, likelihoods, ESS

def observ_gen(N, params, conditional, initial, kernel):
    observ = []
    x = initial(params['initial'])
    for i in range(N+1):
        observ.append(conditional(x, params['conditional']))
        x = kernel(x, params['kernel'])
    return observ