import numpy as np

from distributions.distributions2 import Normal, BetaBinomial, NegativeBinomial, Poisson, Gamma, Binomial, MysteryDistr
from bootstrap.utils_blowflies import transi_s, coeff_r, coeff_beta, proba_r
from functools import partial

class RickerMap(object):

    def __init__(self, p, n0, sigmap, delta, sigmad, tau, m, initial=None, observations=None, length=None, approx='simple'):
        self.p = p
        self.n0 = n0
        self.sigmap = sigmap
        self.delta = delta
        self.sigmad = sigmad
        self.tau = tau
        self.m = m

        if initial:
            self.initial = initial
        else:
            self.initial = Normal().sample(loc=np.array([50], dtype=np.float64), scale=np.array([1], dtype=np.float64))

        self.kernel = self.Kernel(delta=self.delta, sigmad=self.sigmad, p=self.p, n0=self.n0, sigmap =self.sigmap)
        self.prior = self.kernel
        self.conditional = self.Conditional(p=self.p, n0=self.n0, sigmap=self.sigmap, m=self.m, tau=self.tau,
                                            delta=self.delta, sigmad=self.sigmad)

        self.proposal = self.Proposal()


        if observations is not None:
            self.observations = observations
        else:
            self.observations, self.state = self.observ_gen(length)

    class Kernel(object):

        def __init__(self, delta, sigmad, p, n0, sigmap):
            self.distribution = BetaBinomial()
            self.delta = delta
            self.shape = 1/sigmad**2
            self.p = p,
            self.n0 = n0

        def sample(self, ancestor):
            #TODO
            pass

        #@profile
        def density(self, particle, ancestor):
            return [1]*len(particle)

    class Conditional(object):

        def __init__(self, p, n0, sigmap, m, tau, delta, sigmad):
            self.distribution_r = NegativeBinomial()
            self.distribution_s = MysteryDistr
            self.p = p
            self.n0 = n0
            self.shape_r = 1/sigmap**2
            self.m = m
            self.tau = tau
            self.delta = delta
            self.shape_s = 1/sigmad**2
            self.coeff_r = coeff_r
            self.coeff_beta = coeff_beta
            self.transi_s = transi_s
            self.proba_r = proba_r

        def sample(self, ancestor):
            return self.distribution_s.sample(ancestor)

        def density(self, particle, observation, ancestor):
            prev, delayed = zip(*ancestor)
            prev = np.array(prev, dtype=np.float64)
            delayed = np.array(delayed, dtype=np.float64)

            ss = self.sample(prev)
            dens_ss = self.transi_s(ss, prev, self.delta, self.shape_s, len(ss))
            proposal_ss = self.distribution_s.density(ss, prev, self.delta, self.shape_s)
            weight_ss = np.multiply(dens_ss, proposal_ss)

            coeff_beta = self.coeff_beta(delayed, self.p, self.n0, len(observation))
            proba_r = self.proba_r(coeff_beta, self.shape_r)
            rs_total = []
            counts = np.zeros(len(particle))
            for i in range(self.m):
                rs = self.distribution_r.sample(n=np.array([self.shape_r]*len(particle), dtype=np.float64), p=proba_r)
                rs = [rs[i] for i in range(len(rs)) if rs[i] + ss[i] <= observation]
                rs_total.extend(rs)
                for j in range(len(particle)):
                    if rs[i] + ss[i] <= particle:
                        counts[i] += 1
            cond_weights = np.multiply(counts, 1/self.m)

            np.multiply(weight_ss, cond_weights, out=weight_ss)
            return weight_ss

    class Proposal(object):

        def __init__(self):
            pass

        def sample(self, ancestor, observation):
            return ancestor

        #@profile
        def density(self, particle, ancestor, observation):
            return [1]*len(ancestor)

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