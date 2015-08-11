library(rootSolve)
xs = seq(from=-0.11, to=0.2, length.out=1000)
approx = (1+xs)
expo = exp(xs)
plot(expo-approx, type="l")

gammasalpha = rgamma(1000, shape=3, rate=5)
betasalpha = 50000/5*rbeta(10000, shape1=3, shape2=50000)
ks.test(betasalpha, gammasalpha)
qqplot(gammasalpha, betasalpha)
abline(0, 1)


gams = rgamma(10000, shape=10, rate=10)
var = exp(-gams)
approx = rbeta(10000, shape1=6.739281, shape2=10.736471)
hist(var, breaks=40)
hist(approx, breaks=40)
qqplot(var, approx)
abline(0,1)

binom.ori = c()
for (v in var) binom.ori = c(binom.ori, rbinom(1, size=300, p=v))

binom.beta = c()
for (app in approx) binom.beta = c(binom.beta, rbinom(1, size=300, p=app))

hist(binom.ori, breaks=40)
hist(binom.beta, breaks=40)

qqplot(binom.beta, binom.ori)
abline(0,1)

gams = rgamma(10000, shape=10, rate=10)
var = exp(-0.16*gams)
approx = rbeta(10000, shape1=58.23357, shape2=10.01770)
hist(var, breaks=40)
hist(approx, breaks=40)
qqplot(var, approx)
abline(0,1)

func = function(x, alpha, beta) {
  return(log(1-x)*beta^alpha/gamma(alpha)*(-log(x))^(alpha-1)*x^(beta-1))
}
integrate(func, 0, 1, alpha=10, beta=10)

model <- function(x, parms) {
  c(F1 = -digamma(x[1]+x[2])+ digamma(x[1]) + parms[1]/parms[2], 
    F2 = -digamma(x[1]+x[2])+ digamma(x[2]) - integrate(func, 0, 1, alpha=parms[1], beta=parms[2])$value)
}

ss <- multiroot(f = model, parms=c(10, 10), start = c(1, 1))
ss2 = multiroot(f=model, parms=c(10, 10/0.16), start=c(0.01,0.01))

