import numpy as np

from distributions.distributions2 import Normal, BetaBinomial, NegativeBinomial, Poisson, Gamma, Binomial, BetaBinomial
from bootstrap.utils_blowflies import transi_s, coeff_r, coeff_beta, proba_r, calc_delta
from scipy.special import gamma, psi
from scipy.optimize import fsolve
from scipy.integrate import quad

class BlowflyMap(object):

    def __init__(self, p, n0, sigmap, delta, sigmad, tau=14, initial=None, observations=None, length=None, start=0.1, tol=0,
                 *args, **kwargs):
        self.p = p
        self.n0 = n0
        self.sigmap = sigmap
        self.delta = delta
        self.sigmad = sigmad
        self.tau = tau
        self.start = start
        self.tol = tol
        self.coeff = self.coeff_betabinom(1/self.sigmad**2, 1/(self.sigmad**2*self.delta), self.start)
        if self.coeff[0] <= 0 or self.coeff[1] <=0:
            raise Exception

        if initial:
            self.initial = initial
        else:
            self.initial = Normal().sample(loc=np.array([1000], dtype=np.float64), scale=np.array([10], dtype=np.float64))

        self.kernel = self.Kernel(delta=self.delta, sigmad=self.sigmad)
        self.proposal = self.Proposal(delta=self.delta, sigmad=self.sigmad, coeff_betabinom=self.coeff)
        self.conditional = self.Conditional(p=self.p, n0=self.n0, sigmap=self.sigmap, tau=self.tau, tol=self.tol)

        if observations is not None:
            self.observations = observations
        else:
            self.observations, self.state = self.observ_gen(length)

    class Kernel(object):

        def __init__(self, delta, sigmad):
            self.delta = delta
            self.shape = 1/sigmad**2
            self.transi_s = transi_s

        #@profile
        def density(self, particles, observations, index):
            ancestor = observations[index-1]
            return self.transi_s(particles, ancestor, self.delta, self.shape, len(particles))

    class Conditional(object):

        def __init__(self, p, n0, sigmap, tau, tol):
            self.distribution_r = NegativeBinomial()
            self.p = p
            self.n0 = n0
            self.shape_r = 1/sigmap**2
            self.shape_r_arr = None
            self.tau = tau
            self.tol = tol
            self.coeff_r = coeff_r
            self.coeff_beta = coeff_beta
            self.proba_r = proba_r
            self.proposal_r = NegativeBinomial(tweaked=True)
        #@profile
        def density(self, particles, observations, index):
            ancestor = observations[index-self.tau]
            next_obs = int(observations[index])
            coeff_beta = self.coeff_beta(ancestor, self.p, self.n0, len(particles))
            proba_r = self.proba_r(coeff_beta, self.shape_r, len(particles))

            if self.shape_r_arr is None or len(self.shape_r_arr) != len(particles):
                self.shape_r_arr = np.array([self.shape_r]*len(particles), dtype=np.float64)

            rs = self.proposal_r.sample(n=self.shape_r_arr, p=proba_r, next=next_obs)
            dens_rs = self.distribution_r.density(rs, n=self.shape_r_arr, p=proba_r)
            prop_rs = self.proposal_r.density(rs, n=self.shape_r_arr, p=proba_r)
            weights = np.divide(dens_rs, prop_rs)
            deltas =  calc_delta(rs, particles, next_obs, len(rs), tol=self.tol) #[0 if rs[i] + particles[i] != next_obs else 1 for i in range(len(rs))]
            np.multiply(weights, deltas, out=weights)
            return weights

    class Proposal(object):

        def __init__(self,delta, sigmad,  coeff_betabinom):
            self.delta = delta
            self.shape = 1/sigmad**2
            self.coeff_betabinom = coeff_betabinom
            self.distribution = BetaBinomial(tweaked=True)
            self.particles_nb = None

        def sample(self, observations, index, length):
            if self.particles_nb is None or self.particles_nb != length:
                self.particles_nb = length
                self.shape1 = np.array([self.coeff_betabinom[0]]*self.particles_nb, dtype=np.float64)
                self.shape2 = np.array([self.coeff_betabinom[1]]*self.particles_nb, dtype=np.float64)
            ancestor = np.array([observations[index-1]]*length, dtype=np.int32)
            next_obs = np.array([observations[index]]*length, dtype=np.float64)
            return self.distribution.sample(n=ancestor, shape1=self.shape1, shape2=self.shape2, next=next_obs)

        #@profile
        def density(self, particles, observations, index):
            if self.particles_nb is None or self.particles_nb != len(particles):
                self.particles_nb = len(particles)
                self.shape1 = np.array([self.coeff_betabinom[0]]*self.particles_nb, dtype=np.float64)
                self.shape2 = np.array([self.coeff_betabinom[1]]*self.particles_nb, dtype=np.float64)
            ancestor = np.array([observations[index-1]]*len(particles), dtype=np.int32)
            return self.distribution.density(particles, n=ancestor, shape1=self.shape1, shape2=self.shape2)

    def observ_gen(self, length):
        observ = np.empty(length)
        state = np.empty(length)
        epsilon = Gamma().sample(shape=np.array([1/self.sigmad**2]*length, dtype=np.float64),
                                 scale=np.array([self.sigmad**2]*length, dtype=np.float64))
        e = Gamma().sample(shape=np.array([1/self.sigmap**2]*(length-self.tau+1), dtype=np.float64),
                             scale=np.array([self.sigmap**2]*(length-self.tau+1), dtype=np.float64))
        x = self.initial
        for i in range(self.tau):
            observ[i] = state[i] = x[0]
            x = Binomial().sample(n=x, p=np.array([np.exp(-self.delta*epsilon[i])], dtype=np.float64))

        r = Poisson().sample(lam=np.array([self.p*state[i-self.tau+1]*np.exp(-state[i-self.tau+1]/self.n0)*e[i-self.tau+1]], dtype=np.float64))
        for i in range(self.tau, length):
            observ[i] =  state[i] = x[0] + r[0]
            x = Binomial().sample(n=np.array([state[i]], dtype=np.float64), p=np.array([np.exp(-self.delta*epsilon[i])], dtype=np.float64))
            r = Poisson().sample(lam=np.array([self.p*state[i-self.tau+1]*np.exp(-state[i-self.tau+1]/self.n0)*e[i-self.tau+1]], dtype=np.float64))

        return observ, state

    def coeff_betabinom(self, alpha, beta, start):
        def func(x): return np.log(1-x)*(beta**alpha)/gamma(alpha)*(-np.log(x))**(alpha-1)*(x**(beta-1))

        def equations(p): return (-psi(p[0]+p[1])+psi(p[0])+alpha/beta, -psi(p[0]+p[1])+psi(p[1])-quad(func, 0, 1)[0])

        return fsolve(equations, (start, start))
