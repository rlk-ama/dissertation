import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from ricker_gamma import RickerMap

mu = 0
size = 10000
n_prev = 5
y = 7
phi = 10

# lognorm = np.random.lognormal(mean=mean, sigma=sigma, size=size)
# lognorm = sorted(np.multiply(lognorm, r*n_prev*np.exp(-n_prev)))
# alpha, beta = param_gamma(n_prev, r, mean, sigma)
# gamma = sorted(np.random.gamma(shape=alpha, scale=beta, size=size))
#
# fig1 = plt.figure()
# plt.plot(lognorm, gamma)
# plt.show()
# plt.close()


def optimal_improper(n, n_prev, r, phi, sigma, y):
    logout = -phi*n + (y-1)*np.log(n) - 1/(2*sigma**2)*(np.log(n/(r*n_prev)) + n_prev)**2
    try:
        out = np.exp(logout)
    except:
        out = 0
    return out

def initial():
    pass

def compare(type_approx, start, end, length, r_base, sigma_base):
    r, sigma = np.exp(r_base), np.sqrt(sigma_base/10)
    Map_simple = RickerMap(r, phi, mu, sigma, initial, approx=type_approx)
    normalizing = quad(lambda x: optimal_improper(x, n_prev, r, phi, sigma, y), 0, np.inf)[0]
    points = np.linspace(start, end, length)
    approx = [Map_simple.proposal_density(n, n_prev, y) for n in points]
    optimal = [optimal_improper(n, n_prev, r, phi, sigma, y)/normalizing for n in points]
    if approx == 'simple':
        relative = [(approx[i] - optimal[i])/optimal[i] if optimal[i] != 0 else 0 for i in range(len(approx))]
    else:
        relative = [(approx[i] - optimal[i])/optimal[i] if optimal[i] != 0 else 0 for i in range(len(approx))]
    absolute = [(approx[i] - optimal[i]) for i in range(len(approx))]

    fig2 = plt.figure()
    plt.plot(points[:6], optimal[:6])
    plt.plot(points[:6], approx[:6])
    plt.savefig('beginning_%s_%s_%s.pdf' % (type_approx, r_base, sigma_base))
    plt.close()
    fig3 = plt.figure()
    plt.plot(points[0:20], relative[0:20])
    plt.savefig('relative_beginning_%s_%s_%s.pdf' % (type_approx, r_base, sigma_base))
    plt.close()
    fig4 = plt.figure()
    plt.plot(points[:60], absolute[:60])
    plt.savefig('absolute_%s_%s_%s.pdf' % (type_approx, r_base, sigma_base))
    plt.close()
    fig5 = plt.figure()
    plt.plot(points[60:200], relative[60:200])
    plt.savefig('relative_tail_%s_%s_%s.pdf' % (type_approx, r_base, sigma_base))
    plt.close()
    fig6 = plt.figure()
    plt.plot(points[50:60], optimal[50:60])
    plt.plot(points[50:60], approx[50:60])
    plt.savefig('tail_%s_%s_%s.pdf' % (type_approx, r_base, sigma_base))
    plt.close()
    fig7 = plt.figure()
    plt.plot(points[:70], optimal[:70])
    plt.plot(points[:70], approx[:70])
    plt.savefig('global_%s_%s_%s.pdf' % (type_approx, r_base, sigma_base))
    plt.close()

compare('complex', 0, 30, 500, 3, 3)
compare('simple', 0, 30, 500, 3, 3)

compare('complex', 0, 30, 500, 1, 3)
compare('simple', 0, 30, 500, 1, 3)

compare('complex', 0, 30, 500, 3, 1)
compare('simple', 0, 30, 500, 3, 1)