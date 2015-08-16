#travel time prediction using gaussian process regression a trajectory based approach
library(rjags)
library(lattice)
readFiles = function(path, nb, start, end, skip, start_file=0) {
  l = list()
  for (i in start_file:(start_file+nb-1)) {
    d = read.table(paste(path, sprintf("_%i.txt", i), sep=""), skip=skip)
    l[[i+1-start_file]] = mcmc(d[start:end,])
  }
  return(mcmc.list(l))
}

effectiveSizes = function(samples) {
  sizes = matrix(, nrow=0, ncol=ncol(samples[[1]]))
  for (sample in samples) sizes = rbind(effectiveSize(sample), sizes)
  return(sizes)
}

correlations = function(samples) {
  combinations = combn(ncol(samples[[1]]), 2)
  correlations = matrix(, nrow=0, ncol=ncol(combinations))
  for (sample in samples) {
    cors = apply(combinations, 2, function(x) cor(sample[, x])[1,2])
    correlations = rbind(cors, correlations)
  }
  return(correlations)
}

runningMeans = function(samples, index, bounds, nb){
  plot(cumsum(samples[[1]][, index])/seq(along=samples[[1]][, index]),
       type='l', ylab="Running mean", col=1, ylim=bounds)
  for (j in 2:nb) {
    lines(cumsum(samples[[j]][, index])/seq(along=samples[[j]][, index]),
          col=j)
  }
}

measure = function(samples, true) {
  means = sapply(samples, function(x) apply(x, 2, mean)) - true
  coeff = apply(means^2, 2, function(x) prod(x)^(1/length(x)))
  return(coeff)
}


mse = function(samples, true) {
  means = sapply(samples, function(x) apply(x, 2, mean)) - true
  coeff = apply(means^2, 1, median)
  return(coeff)
}

NB = 51
true = c(44.7, 10, 0.3)
test2 = readFiles("/home/raphael/runs_prior/samples", NB, 5000, 17500, skip=1)
sizes2 = effectiveSizes(test2)
correlations2 = correlations(test2)

chains <- sample(35, 5)
densityplot(test2[chains][, 1], aspect=1)
densityplot(test2[chains][, 2], aspect=1)
densityplot(test2[chains][, 3], aspect=1)

traceplot(test2[[1]][, 1])
traceplot(test2[[1]][, 2])
traceplot(test2[[1]][, 3])

acf(test2[[1]][, 1], lag=400, main="")
acf(test2[[1]][, 2], lag=400, main="")
acf(test2[[1]][, 3], lag=400, main="")

runningMeans(test2, 1, c(20, 60), NB)
runningMeans(test2, 2, c(8, 12), NB)
runningMeans(test2, 3, c(0, 0.7), NB)

for (i in 1:3) {
  gelman.plot(test2[, i])
}

geweke.results <- sapply(test2, function(x) geweke.diag(x, frac1 = 0.1, frac2 = 0.5)$z)
p.values <- apply(geweke.results, 2, function(x) 1-pnorm(x))
p.values.adjust <- apply(p.values, 1, function(x) p.adjust(x, "fdr"))

summary(test2)
summary(measure(test2, true))
mse(test2, true)

NB=51
test3 = readFiles("/home/raphael/dissertation/ricker-libbi/samples", NB, 5000, 17500, skip=0)
traceplot(test3[[1]][, 1])
traceplot(test3[[1]][, 2])
traceplot(test3[[1]][, 3])

runningMeans(test3, 1, c(20, 60), NB)
runningMeans(test3, 2, c(8, 12), NB)
runningMeans(test3, 3, c(0, 0.7), NB)


summary(test3)
summary(measure(test3, true))
mse(test3, true)


NB=51
true = c(44.7, 10, 1, 0.3)
test4 = readFiles("/home/raphael/dissertation/ricker-libbi/samples_generalized", NB, 5000, 17500, skip=0)
traceplot(test4[[40]][, 1])
traceplot(test4[[40]][, 2])
traceplot(test4[[40]][, 3])
traceplot(test4[[40]][, 4])

runningMeans(test4, 1, c(20, 60), NB)
runningMeans(test4, 2, c(8, 12), NB)
runningMeans(test4, 3, c(0.5, 1.5), NB)
runningMeans(test4, 4, c(0, 0.7), NB)


summary(test4)
summary(measure(test4, true))
mse(test4, true)

NB = 49
test5 = readFiles("/home/raphael/runs_prior_generalized_long/samples", NB, 5000, 17500, skip=1)
traceplot(test5[[1]][, 1])
traceplot(test5[[1]][, 2])
traceplot(test5[[1]][, 3])
traceplot(test5[[1]][, 4])

runningMeans(test5, 1, c(20, 60), NB)
runningMeans(test5, 2, c(8, 12), NB)
runningMeans(test5, 3, c(0.5, 1.5), NB)
runningMeans(test5, 4, c(0, 0.7), NB)

summary(test5)
summary(measure(test5, true))
mse(test5, true)

NB = 35
test6 = readFiles("/home/raphael/runs_posterior_generalized_long/samples", NB, 5000, 17500, skip=1,
                  start_file=12)
traceplot(test6[[3]][, 1])
traceplot(test6[[3]][, 2])
traceplot(test6[[3]][, 3])
traceplot(test6[[3]][, 4])

runningMeans(test6, 1, c(20, 60), NB)
runningMeans(test6, 2, c(8, 12), NB)
runningMeans(test6, 3, c(0.5, 1.5), NB)
runningMeans(test6, 4, c(0, 0.7), NB)

summary(test6)
summary(measure(test6, true))
mse(test6, true)

acf_prior = readFiles("/home/raphael/test_acf_prior/samples", 1, 5000, 10000, skip=1)
acf_gamma = readFiles("/home/raphael/test_acf_gamma/samples", 1, 5000, 10000, skip=1)
acf(acf_prior[[1]][, 1], lag=100)
acf(acf_gamma[[1]][, 1], lag=100)

acf(acf_prior[[1]][, 2], lag=100)
acf(acf_gamma[[1]][, 2], lag=100)

acf(acf_prior[[1]][, 3], lag=100)
acf(acf_gamma[[1]][, 3], lag=100)

effectiveSizes(acf_prior)
effectiveSizes(acf_gamma)


NB = 1
test_opti = readFiles("/home/raphael/test_new/samples", NB, 5000, 17500, skip=1)
test_prior = readFiles("/home/raphael/test_new_prior/samples", NB, 5000, 17500, skip=1)
acf(test_opti[[1]][,1], lag=130, main="")
acf(test_prior[[1]][,1], lag=130, main="")
acf(test_opti[[1]][,2], lag=100, main="")
acf(test_prior[[1]][,2], lag=100, main="")
acf(test_opti[[1]][,3], lag=70, main="")
acf(test_prior[[1]][,3], lag=70, main="")
effectiveSizes(test_opti)
effectiveSizes(test_prior)

NB = 10
lagopus = readFiles("/home/raphael/salmon1/samples", NB, 5000, 17500, skip=1)
traceplot(lagopus[[1]][, 1])
traceplot(lagopus[[1]][, 2])
traceplot(lagopus[[1]][, 3])
traceplot(lagopus[[1]][, 4])
effectiveSize(lagopus[[1]])

runningMeans(lagopus, 1, c(3, 8), NB)
runningMeans(lagopus, 2, c(8, 12), NB)
runningMeans(lagopus, 3, c(1, 1.4), NB)
runningMeans(lagopus, 4, c(8, 12), NB)

for (i in 1:4) {
  gelman.plot(lagopus[, i])
}

geweke.results <- sapply(lagopus, function(x) geweke.diag(x, frac1 = 0.1, frac2 = 0.5)$z)
p.values <- apply(geweke.results, 2, function(x) 1-pnorm(x))
p.values.adjust <- apply(p.values, 1, function(x) p.adjust(x, "fdr"))



salmon = readFiles("/home/raphael/salmon1/samples", 1, 0, 17500, skip=1)
traceplot(salmon[[1]][, 1])
traceplot(salmon[[1]][, 2])
traceplot(salmon[[1]][, 3])
traceplot(salmon[[1]][, 4])

#python3 examples/ricker_pmmh_stability.py --iterations 17500 --burnin 2500 --adaptation 2500 --destination /home/raphael/lagopus/ --observations /home/raphael/lagopus_obs.txt --particles 500 --chains 1 --number 77 --sigma_init 0.6 --r_init 1.2 --scaling_init 600 --proposal_scaling 300 --proposal_phi 2 --phi_init 5 --scaling_model True --particle_init 125 --proposal_r 1
NB = 3
lagopus = readFiles("/home/raphael/lagopus/samples", NB, 5000, 17500, skip=1)
traceplot(lagopus[[3]][, 1], ylim=c(1.2/2, 1.5*1.2))
traceplot(lagopus[[3]][, 2], ylim=c(5/2, 1.5*5))
traceplot(lagopus[[3]][, 3],  ylim=c(0.6/2, 1.5*0.6))
traceplot(lagopus[[3]][, 4],  ylim=c(600/2, 1.5*600))

acf(lagopus[[3]][, 1], lag=100, main="")
acf(lagopus[[3]][, 2], lag=130, main="")
acf(lagopus[[3]][, 3], lag=100, main="")
acf(lagopus[[3]][, 4], lag=100, main="")

runningMeans(lagopus, 1, c(1.2, 1.8), NB)
runningMeans(lagopus, 2, c(4, 7), NB)
runningMeans(lagopus, 3, c(0.4, 0.7), NB)
runningMeans(lagopus, 4, c(400, 700), NB)

summary(lagopus)

r = 1.4795
phi = 5.5827
sigma = 0.6047
K = 576.81
data = as.numeric(read.csv("/home/raphael/lagopus_obs.txt", sep=" ", header=F))
start = as.numeric(read.csv("/home/raphael/lagopus_obs.txt", sep=" ", header=F))[1]
state_func = function(prev, r, sigma) return(r*prev*exp(-prev/K)*rlnorm(1, sdlog=sigma))
obs_func = function(state, phi) return(rpois(1, lambda=phi*state))
obs = c()
state = start
for (i in 1:77) {
  state = state_func(state, r=r, sigma=sigma)
  obs = c(obs, obs_func(state, phi=phi))
}

plot(obs, type="l")
lines(data)

blow = readFiles("/home/raphael/blowfly2/samples", 1, 0, 3000, skip=1)
traceplot(blow[[1]][, 1])
traceplot(blow[[1]][, 2])
traceplot(blow[[1]][, 3])
traceplot(blow[[1]][, 4])
traceplot(blow[[1]][, 5])
summary(blow)
effectiveSizes(blow)
acf(blow[[1]][, 1], lag=200)
acf(blow[[1]][, 2], lag=200)
acf(blow[[1]][, 3], lag=200)
acf(blow[[1]][, 4], lag=200)
acf(blow[[1]][, 5], lag=200)


neww = readFiles("/home/raphael/test_new/samples", 1, 5000, 17500, skip=1)
traceplot(neww[[1]][, 1])
summary(neww)
