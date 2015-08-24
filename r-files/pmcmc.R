library(rjags)
library(lattice)
library(ggmcmc)
library(reshape2)
readFiles = function(path, nb, start, end, skip, start_file=0) {
  l = list()
  for (i in start_file:(start_file+nb-1)) {
    d = read.table(paste(path, sprintf("_%i.txt", i), sep=""), skip=skip)
    l[[i+1-start_file]] = mcmc(d[start:end,])
  }
  return(mcmc.list(l))
}

readFilesCompare = function(path, nb, start, end, skip, start_file=0) {
  l.prior = list()
  l.optimal = list()
  for (i in start_file:(start_file+nb-1)) {
    d = read.table(paste(path, sprintf("_%i_prior.txt", 2*i), sep=""), skip=skip)
    l.prior[[i+1-start_file]] = mcmc(d[start:end,])
    d = read.table(paste(path, sprintf("_%i_optimal.txt", 2*i+1), sep=""), skip=skip)
    l.optimal[[i+1-start_file]] = mcmc(d[start:end,])
  }
  return(list(prior=mcmc.list(l.prior), optimal=mcmc.list(l.optimal)))
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

runningMeans = function(samples, index, name){
  last = c()
  data = cumsum(samples[[1]][, index])/seq(along=samples[[1]][, index])
  last = c(last, data[length(data)])
  dataF = data.frame(x=1:length(data))
  dataF[paste(1)] = data
  for (j in 2:length(samples)) {
    data = cumsum(samples[[j]][, index])/seq(along=samples[[j]][, index])
    last = c(last, data[length(data)])
    dataF[paste(j)] = data
  }
  dataFrameLong = melt(dataF, id.vars="x")
  ggplot(data=dataFrameLong, aes(x=x, y=value, colour=variable)) + 
         ylim(mean(last)/2, mean(last)*1.5) + xlab("Iteration") +
         ylab("Running Mean") + geom_line() + theme(legend.position = "none")
  ggsave(name, width=8, height=6)
}

preprocess = function(samples, func) {
  for (i in 1:length(samples)) {
    for (j in 1:ncol(samples[[i]])) {
      samples[[i]][, j] = func(samples[[i]][, j])
    }
  }
  return(samples)
}

measure = function(samples, true) {
  samples2 = samples
  samples2 = preprocess(samples2, log)
  means = sapply(samples2, function(x) apply(x, 2, mean)) - true
  coeff = apply(means^2, 2, function(x) prod(x)^(1/length(x)))
  return(coeff)
}


mse = function(samples, true) {
  samples2 = samples
  samples2 = preprocess(samples2, log)
  means = sapply(samples2, function(x) apply(x, 2, mean)) - true
  coeff = apply(means^2, 1, median)
  return(coeff)
}

summaryplots = function(samples, parameters, name) {
  data = ggs(samples)
  for (param in parameters) {
    ggs_traceplot(data, family=param) + ylab("Parameter value") + xlab("Iteration")
    ggsave(sprintf("/home/raphael/dissertation/figures/trace%s%s.pdf", name, param),
           width=8, height=6)
    signi = qnorm((1 + 0.95)/2)/sqrt(sum((samples[, grep(param, colnames(samples))])))
    ggs_autocorrelation(data, nLags=100, family=param) + ylim(-0.1, 1) +
      geom_abline(intercept=signi, slope=0, colour="red")
    ggsave(sprintf("/home/raphael/dissertation/figures/acf%s%s.pdf", name, param),
           width=8, height=6)
  }
}

compareAcf = function(samples, samplesPrior, parameters) {
  
}

summaryplotsfull = function(fullsamples, idx, parameters, name) {
  colsnb = ncol(fullsamples[[1]])
  for (i in 1:colsnb) {
    runningMeans(fullsamples, i,
                 sprintf("/home/raphael/dissertation/figures/running%s%s.pdf",
                         name, i))
    gelmed = gelman.plot(fullsamples[, i])$shrink[,, "median"]
    gelhigh = gelman.plot(fullsamples[, i])$shrink[,, "97.5%"]
    x = as.numeric(names(gelhigh))
    dataF = data.frame(x=x, med=gelmed, high=gelhigh)
    ggplot(data=dataF, aes(x=x, y=med)) + geom_line() + 
      geom_line(data=dataF, aes(x=x, y=high), typeline="dashed", colour="red") +
      xlab("Iterations") + ylab("Gelman-Rubin coefficient")
    ggsave(sprintf("/home/raphael/dissertation/figures/gelman%s%s.pdf", name, i), width=8,
           height=6)
  }
  summaryplots(fullsamples[[idx]], parameters, name)
  chains <- sample(length(fullsamples), 3)
  data = ggs(fullsamples[chains])
  for (param in parameters) {
    ggs_density(data, family=param) + xlab("Parameter value") + ylab("Density estimate") +
      theme(legend.position = "none")
    ggsave(sprintf("/home/raphael/dissertation/figures/densities%s%s.pdf", name, param), width=8,
           height=6)
  }
}

NB = 5
true = c(log(44.7), log(10), log(0.3))
sameRicker = readFiles("/home/raphael/ricker-samples-same/samples", NB, 5000, 17500, skip=1)
sizes2 = effectiveSizes(sameRicker)
correlations2 = correlations(sameRicker)
summaryplotsfull(sameRicker, 3, c("V1", "V2", "V3"), "RickerSame")

geweke.results <- sapply(test2, function(x) geweke.diag(x, frac1 = 0.1, frac2 = 0.5)$z)
p.values <- apply(geweke.results, 2, function(x) 1-pnorm(x))
p.values.adjust <- apply(p.values, 1, function(x) p.adjust(x, "fdr"))

summary(sameRicker)
summary(measure(sameRicker, true))
mse(sameRicker, true)

sameRickerPrior = readFiles("/home/raphael/ricker-samples-same-prior/samples", NB, 5000, 17500, skip=1)
diff = (effectiveSizes(sameRicker) - effectiveSizes(sameRickerPrior))/effectiveSizes(sameRicker)
apply(diff, 2, mean)

NB=50
rickerLibBi = readFiles("/home/raphael/dissertation/ricker-libbi/samples", NB, 5000, 17500, skip=0)

summary(rickerLibBi)
summary(measure(rickerLibBi, true))
mse(rickerLibBi, true)


NB=50
true = c(44.7, 10, 1, 0.3)
test4 = readFiles("/home/raphael/dissertation/ricker-libbi/samples_generalized", NB, 5000, 17500, skip=0)

summary(test4)
summary(measure(test4, true))
mse(test4, true)

NB = 50
rickerGamma= readFiles("/home/raphael/ricker-samples/samples",
                  NB, 5000, 17500, skip=1, start_file=12)
summaryplots(test5[[8]], c("V1", "V2", "V3"))
runningMeans(test5, 1, c(20, 60), NB)
runningMeans(test5, 2, c(8, 12), NB)
runningMeans(test5, 3, c(0, 0.7), NB)

summary(test5)
summary(measure(rickerGamma, true))
mse(rickerGamma, true)

NB = 50
test6 = readFiles("/home/raphael/ricker-samples/samples", NB, 5000, 17500, skip=1,
                  start_file=12)

summaryplots(test6[[8]])
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
lagopus3 = readFiles("/home/raphael/lagopus/samples", NB, 0, 17500, skip=1)
summaryplotsfull(lagopus, 3, c("V1", "V2", "V3", "V4"), "Lagopus")

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

blow = readFiles("/home/raphael/blowfly-samples/samples", 1, 0, 5000, skip=1)
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


lagopus3 = readFiles("/home/raphael/tetrao-samples/samples", 1, 10000, 17500, skip=1)
traceplot(lagopus3[[1]][, 1])
traceplot(lagopus3[[1]][, 2])
traceplot(lagopus3[[1]][, 3])
traceplot(lagopus3[[1]][, 4])

acf(lagopus3[[1]][, 1], lag=100)
acf(lagopus3[[1]][, 2], lag=100)
acf(lagopus3[[1]][, 3], lag=100)
acf(lagopus3[[1]][, 4], lag=100)

#python3 examples/ricker_pmmh_stability.py --iterations 17500 --burnin 2500 --adaptation 0 --destination /home/raphael/caradrina-samples/ --observations /home/raphael/caradrina_obs.txt --particles 500 --chains 1 --number 23 --sigma_init 0.2 --r_init 2 --scaling_init 40 --proposal_scaling 10 --proposal_phi 5 --phi_init 10 --scaling_model True --particle_init 10 --proposal_r 1
caradrina = readFiles("/home/raphael/caradrina-samples/samples", 1, 5000, 17500, skip=1)
traceplot(caradrina[[1]][, 1])
traceplot(caradrina[[1]][, 2])
traceplot(caradrina[[1]][, 3])
traceplot(caradrina[[1]][, 4])

acf(caradrina[[1]][, 1], lag=100)
acf(caradrina[[1]][, 2], lag=100)
acf(caradrina[[1]][, 3], lag=100)
acf(caradrina[[1]][, 4], lag=100)

summary(caradrina)


compare = readFilesCompare("/home/raphael/ricker-samples-compare/samples", 10, 5000, 17500, skip=1)
compare.prior = compare$prior
compare.optimal = compare$optimal
rel = (effectiveSizes(compare.optimal)-effectiveSizes(compare.prior))/effectiveSizes(compare.prior)
mean.var = apply(rel, 2, mean)
mean.tot = mean(mean.var)
