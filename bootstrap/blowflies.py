import numpy as np

from distributions.distributions2 import Normal, BetaBinomial, NegativeBinomial, Poisson, Gamma, Binomial, BetaBinomial
from bootstrap.utils_blowflies import transi_s, coeff_r, coeff_beta, proba_r
from scipy.special import gamma, psi
from scipy.optimize import fsolve
from scipy.integrate import quad

class BlowflyMap(object):

    def __init__(self, p, n0, sigmap, delta, sigmad, tau, m, initial=None, observations=None, length=None):
        self.p = p
        self.n0 = n0
        self.sigmap = sigmap
        self.delta = delta
        self.sigmad = sigmad
        self.tau = tau
        self.m = m
        self.coeff = self.coeff_betabinom(self.sigmad)

        if initial:
            self.initial = initial
        else:
            self.initial = Normal().sample(loc=np.array([50], dtype=np.float64), scale=np.array([1], dtype=np.float64))

        self.kernel = self.Kernel(delta=self.delta, sigmad=self.sigmad, choose=self.choose)
        self.proposal = self.Proposal(delta=self.delta, sigmad=self.sigmad, choose=self.choose, coeff_betabinom=self.coeff)
        self.conditional = self.Conditional(p=self.p, n0=self.n0, sigmap=self.sigmap, m=self.m, tau=self.tau, choose=self.choose)

        if observations is not None:
            self.observations = observations
        else:
            self.observations, self.state = self.observ_gen(length)

    class Kernel(object):

        def __init__(self, delta, sigmad, choose):
            self.delta = delta
            self.shape = 1/sigmad**2
            self.choose = choose
            self.transi_s = transi_s

        #@profile
        def density(self, particles, indices, observations):
            ancestors = self.choose(indices, observations, 1)
            return self.transi_s(particles, ancestors, self.delta, self.shape, len(particles))

    class Conditional(object):

        def __init__(self, p, n0, sigmap, m, tau, choose):
            self.distribution_r = NegativeBinomial()
            self.p = p
            self.n0 = n0
            self.shape_r = 1/sigmap**2
            self.shape_r_arr = None
            self.m = m
            self.tau = tau
            self.coeff_r = coeff_r
            self.coeff_beta = coeff_beta
            self.proba_r = proba_r
            self.choose = choose
            self.proposal_r = NegativeBinomial(tweaked=True)

        def sample(self, ancestors, n, p):
            return self.distribution_r.sample(ancestors, n=n, p=p)

        def density(self, indices, particles, observations, next):
            ancestors = self.choose(indices, observations, self.tau)
            coeff_beta = self.coeff_beta(ancestors, self.p, self.n0, len(ancestors))
            proba_r = self.proba_r(coeff_beta, self.shape_r)

            if self.shape_r_arr == None:
                self.shape_r_arr = np.array([self.shape_r]*len(ancestors), dtype=np.float64)

            rs = self.distribution_r.sample(n=self.shape_r_arr, p=proba_r)
            dens_rs = self.distribution_r.density(rs, n=self.shape_r_arr, p=proba_r)
            prop_rs = self.proposal_r.density(rs, n=self.shape_r_arr, p=proba_r, next=next)
            weights = np.divide(dens_rs, prop_rs)
            deltas =  [0 if rs[i] + particles[i] <= next else 0 for i in range(len(rs))]
            np.multiply(weights, deltas, out=weights)
            return weights

    class Proposal(object):

        def __init__(self,delta, sigmad, choose, coeff_betabinom):
            self.delta = delta
            self.shape = 1/sigmad**2
            self.choose = choose
            self.coeff_betabinom = coeff_betabinom
            self.distribution = BetaBinomial(tweaked=True)
            self.particles_nb = None

        def sample(self, indices, observations, observation_next):
            if self.particles_nb == None:
                self.particles_nb = len(indices)
                self.coeff_betabinom = np.array([self.coeff_betabinom]*self.particles_nb, dtype=np.float64)
            ancestors = self.choose(indices, observations, 1)
            return self.distribution.sample(n=ancestors, shape1=self.coeff_betabinom, shape2=self.coeff_betabinom,
                                            next=observation_next)

        #@profile
        def density(self, particles, indices, observations):
            if self.particles_nb == None:
                self.particles_nb = len(indices)
                self.coeff_betabinom = np.array([self.coeff_betabinom]*self.particles_nb, dtype=np.float64)
            ancestors = self.choose(indices, observations, 1)
            return self.distribution.density(particles, n=ancestors, shape1=self.coeff_betabinom, shape2=self.coeff_betabinom)

    def observ_gen(self, length):
        observ = np.empty(length)
        state = np.empty(length)
        epsilon = Gamma().sample(shape=np.array([1/self.sigmad**2]*length, dtype=np.float64),
                                 scale=np.array([self.sigmad**2]*length, dtype=np.float64))
        e = Gamma().sample(shape=np.array([1/self.sigmap**2]*(length-self.tau), dtype=np.float64),
                             scale=np.array([self.sigmap**2]*(length-self.tau), dtype=np.float64))
        x = self.initial
        for i in range(self.tau):
            observ[i] = state[i] = x[0]
            x = Binomial().sample(n=x, p=np.array([np.exp(-self.delta*epsilon[i])], dtype=np.float64))

        r = Poisson().sample(lam=np.array(self.p*state[i-self.tau]*np.exp(-state[i-self.tau]/self.n0)*e[i]), dtype=np.float64)
        for i in range(self.tau, length):
            observ[i] =  state[i] = x[0] + r[0]
            x = Binomial().sample(n=x, p=np.array([np.exp(-self.delta*epsilon[i])], dtype=np.float64))
            r = Poisson().sample(lam=np.array(self.p*state[i-self.tau]*np.exp(-state[i-self.tau]/self.n0)*e[i]), dtype=np.float64)

        return observ, state

    def choose(self, indices, observations, delay):
        row = observations[-delay]
        return np.take(row, indices)

    def coeff_betabinom(self, alpha):
        def func(x): return np.log(1-x)*(alpha**alpha)/gamma(alpha)*(-np.log(x))**(alpha-1)*(x**(alpha-1))

        def equations(p): return (-psi(p[0]+p[1])+psi(p[0])+1, -psi(p[0]+p[1])+psi(p[1])-quad(func, 0, 1)[0])

        return fsolve(equations, (alpha, alpha))
