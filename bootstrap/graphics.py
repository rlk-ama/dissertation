import numpy as np
import matplotlib.pyplot as plt


from bootstrap.ricker_gamma import RickerMap
from bootstrap.filter import BootstrapFilter
from bootstrap.kalman import KalmanMap
from distributions.distributions import Gamma

shape = 3
scale = 1
r = 16
phi = 5
sigma = 0.5
mu = 0
NOS = 40
NBS = 5000
Ns = [10*i for i in range(1, 50)]

# def initial_ricker():
#     out = np.random.gamma(shape=shape, scale=scale)
#     return out

initial_ricker = Gamma(func_shape=lambda args: shape, func_scale=lambda args: scale)


Map_ricker = RickerMap(r, phi, sigma, length=NOS, initial=initial_ricker, approx="simple")
#Map_gamma = KalmanMap(0.9, np.sqrt(0.4), np.sqrt(0.6), initial_gamma)


filter = BootstrapFilter(0, NOS, NBS, Map_ricker, proposal={'prior': False, 'optimal': True})

proposal, estim, likeli, ESS = next(filter.filter())


# idx = [ESS.index(elem) for elem in sorted(ESS)[:3]]
#
# for i in idx:
#     samples = np.random.choice(NBS, 50, replace=False)
#     op = 0
#     print(ESS[i])
#     for sample in samples:
#         proposed = np.array([Map_ricker.proposal(estim[i-1][sample], filter.observations[i]) for _ in range(1000)])[:, np.newaxis]
#         hist, bins = np.histogram(proposed, bins=20, density=True)
#         widths = np.diff(bins)
#         plt.bar(bins[:-1], hist, widths, alpha=op)
#         op += 1/50
#         # kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(proposed)
#         # pts = np.linspace(0, 6, 1000)[:, np.newaxis]
#         # log_dens = kde.score_samples(pts)
#         # plt.plot(pts, np.exp(log_dens))
#     plt.show()

# estim_all = []
# likeli_all = []
# ESS_all = []
# proposals_all = []
#
# for proposal, estim, likeli, ESS in filter.filter():
#     proposals_all.append(proposal)
#     estim_all.append(estim)
#     likeli_all.append(likeli)
#     ESS_all.append(ESS)
# #
# #
# ESS_norm = [[ess/Ns[i] for ess in ESS_all[i]] for i in range(len(ESS_all))]
# ESS_t = []
# for i in range(len(ESS_all[0])):
#     ESS_t.append([ess[i] for ess in ESS_norm])
#
# for i in range(NOS):
#     plt.ylim((0,1.2))
#     plt.plot([i for i in range(len(Ns))], ESS_t[i])
#     plt.show()



#fig, ax1 = plt.subplots()
#fig1 = plt.figure()
#for i in range(len(estim_all)):
    #mean_esti = [np.mean(est) for est in estim_all[i]]
# #ax1.plot([i for i in range(NOS)], mean_esti)
    #plt.plot([i for i in range(NOS)], mean_esti)
#ax1.plot([i for i in range(NOS+1)], filter.observations)
    #plt.plot([i for i in range(NOS+1)], filter.state)
# #plt.plot([i for i in range(NOS+1)], filter.observations)
#     plt.savefig('diagno_%s.pdf' % proposals_all[i])
#     plt.close()
#     plt.plot([i for i in range(NOS)], ESS_all[i])
#     plt.ylim((0, NBS))
#     plt.savefig("ESS_%s.pdf" % proposals_all[i])
#     plt.close()
mean_esti = [np.mean(est) for est in estim]
plt.plot([i for i in range(NOS)], mean_esti)
plt.plot([i for i in range(NOS+1)], Map_ricker.state)
plt.show()
#ax2 = ax1.twinx()
#ax2.plot([i for i in range(NOS)], ESS, color="red")
# plt.savefig('ESS_gamma.pdf')
#plt.show()
# plt.close()
#fig3 = plt.figure()
#plt.plot(rs, liks)
#plt.plot(rs, liks2)
#plt.show()
#plt.plot([i for i in range(NOS)], likeli)
#plt.plot([i for i in range(NOS)], likeli_prior)
# plt.savefig('loglik_gamma.pdf')
# plt.close()
#plt.show()
#mean_esti = [np.mean(est) for est in estim_all[0]]
#mean_esti_prior = [np.mean(est) for est in estim_all[1]]
#fig2 = plt.figure()
#plt.plot([i for i in range(NOS)], mean_esti)
#plt.plot([i for i in range(NOS)], mean_esti_prior)
#plt.plot([i for i in range(NOS+1)], filter.state)
#plt.show()
#plt.plot([i for i in range(NOS+1)], state)
# plt.savefig('diagno_prior.pdf')
# plt.close()
#fig2 = plt.figure()
#plt.plot([i for i in range(NOS)], ESS_all[0])
#print(proposals_all[0])
plt.plot([i for i in range(NOS)], ESS)
#plt.plot([i for i in range(NOS)], ESS_all[1])
#print(proposals_all[1])
# plt.plot([i for i in range(NOS)], ESS)
# plt.savefig("ESS_gamma.pdf")
# plt.close()
# plt.ylim((0, NBS))
# plt.ylim((0, 5000))
#plt.savefig('ESS_gamma.pdf')
#plt.close()
# fig3 = plt.figure()
#plt.plot([i for i in range(NOS)], likeli2)
# plt.savefig('loglik_prior.pdf')
# plt.close()
#plt.plot([i for i in range(NOS)], likeli_all[0])
#plt.plot([i for i in range(NOS)], likeli_all[1])
plt.show()