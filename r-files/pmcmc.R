#travel time prediction using gaussian process regression a trajectory based approach
library(rjags)
library(lattice)
readFiles = function(path, nb, start, end) {
  l = list()
  for (i in 0:(nb-1)) {
    d = read.table(paste(path, sprintf("_%i.txt", i), sep=""))
    l[[i+1]] = mcmc(d[start:end,])
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
  plot(cumsum(samples[[1]][, index])/seq(along=test2[[1]][, index]),
       type='l', ylab="Running mean", col=1, ylim=bounds)
  for (j in 2:nb) {
    lines(cumsum(samples[[j]][, index])/seq(along=samples[[j]][, index]),
          col=j)
  }
}

NB = 5
test = readFiles("/home/raphael/samples", NB)
sizes = effectiveSizes(test)
correlations = correlations(test)

NB = 10
test2 = readFiles("/home/raphael/samples_long_repeat2", NB, 2000, 8000)
sizes2 = effectiveSizes(test2)
correlations2 = correlations(test2)

chains <- sample(10, 3)
densityplot(test2[chains][, 1], aspect=1)
densityplot(test2[chains][, 2], aspect=1)
densityplot(test2[chains][, 3], aspect=1)

traceplot(test2[[5]][, 1])
traceplot(test2[[5]][, 2])
traceplot(test2[[5]][, 3])

acf(test2[[1]][, 1], lag=40, main="")
acf(test2[[1]][, 2], lag=40, main="")
acf(test2[[1]][, 3], lag=40, main="")

runningMeans(test2, 1, c(20, 50), NB)
runningMeans(test2, 2, c(8, 12), NB)
runningMeans(test2, 3, c(0, 0.7), NB)

for (i in 1:3) {
  gelman.plot(test2[, i])
}

geweke.results <- sapply(test2, function(x) geweke.diag(x, frac1 = 0.1, frac2 = 0.5)$z)
p.values <- apply(geweke.results, 2, function(x) 1-pnorm(x))
p.values.adjust <- apply(p.values, 1, function(x) p.adjust(x, "fdr"))
