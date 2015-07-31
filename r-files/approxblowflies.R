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

func = function(x) {
  return(log(1-x)*10^10/gamma(10)*(-log(x))^9*x^9)
}
integrate(func, 0, 1)

model <- function(x) c(F1 = -digamma(x[1]+x[2])+ digamma(x[1]) + 1, 
                       F2 = -digamma(x[1]+x[2])+ digamma(x[2]) - integrate(func, 0, 1)$value)

ss <- multiroot(f = model, start = c(1, 1))

