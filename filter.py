import numpy
from matplotlib import pyplot, collections
from mpl_toolkits.mplot3d import Axes3D

def normalize(l):
    total = sum(l)
    return [item/total for item in l]

def filter(param=None, start=None, end=None, N=None, kernel=None, conditional=None, initial_density=None, observations=None):
    particles_all = numpy.empty(shape=(end-start, N))
    particles = []
    weights = []
    likelihoods = []

    for i in range(N):
        particles.append(initial_density(param['initial_density']))
        weights.append(1.0/N)
        likelihood = 1.0/N*sum(weights)

    for i in range(start+1, end):
        weights_norm = normalize(weights)
        indices = numpy.random.multinomial(N, weights_norm, size=1)[0]
        ancestors = [particles[j] for j in range(len(indices)) for _ in range(indices[j])]
        particles_all[i-1] = ancestors
        particles = [kernel(ancestor, i, param['kernel']) for ancestor in ancestors]
        weights = [conditional(particle, observations[i-1], param['conditional']) for particle in particles]
        likelihood *= 1.0/N*sum(weights)
        likelihoods.append(likelihood)

    return particles_all, likelihoods


def kernel1(x, t, param):
    mean = 1/2*x + 25*x/(1+x**2) + 8*numpy.cos(1.2*t) + param['mean']
    out = numpy.random.normal(mean, param['sd'])
    return out

def conditional1(x, y, param):
    mean = x**2/20 + param['mean']
    out = 1/(numpy.sqrt(2*numpy.pi)*param['sd'])*numpy.exp(-1/(2*param['sd']**2)*(y - mean)**2)
    return out

def conditional1_draw(x, param):
    mean = x**2/20 + param['mean']
    out = numpy.random.normal(mean, param['sd'])
    return out

def initial_density1(param):
    out = numpy.random.normal(param['mean'], param['sd'])
    return out

params = {
    'initial_density': {'mean': 0.0, 'sd': numpy.sqrt(10.0)},
    'kernel': {'mean': 0.0, 'sd': numpy.sqrt(10.0)},
    'conditional': {'mean': 0.0, 'sd': 1.0},
    }

obs = []
x = initial_density1(params['initial_density'])
for i in range(1, 100):
    obs.append(conditional1_draw(x, params['conditional']))
    x = kernel1(x, i, params['kernel'])

estim, likeli = filter(param=params, start=0, end=100, N=1000, kernel=kernel1, conditional=conditional1, initial_density=initial_density1,
                       observations=obs)

pyplot.plot([i for i in range(1,100)], likeli)
pyplot.show()
hists = [numpy.histogram(estimation, bins=10) for estimation in estim]
lines = [zip(hist[1][:-1], [elem/1000.0 for elem in hist[0]]) for hist in hists][0::10]
steps = collections.LineCollection(lines)
fig = pyplot.figure()
ax = fig.gca(projection='3d')
ax.add_collection3d(steps, zs=[i for i in range(1,11)], zdir='y')
ax.set_xlabel('X')
ax.set_xlim3d(-25, 25)
ax.set_ylabel('Y')
ax.set_ylim3d(0, 10)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 1)
pyplot.show()