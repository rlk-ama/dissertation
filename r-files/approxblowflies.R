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
alpha = 1/0.1
beta = alpha/0.16
ss2 = multiroot(f=model, parms=c(alpha, beta), start=c(0.1, 0.1))

model2 <- function(x, parms) {
  c(F1 = -log(x[1]+x[2]) + 1/(2*(x[1]+x[2])) + log(x[1]) - 1/(2*x[1]) + parms[1]/parms[2], 
    F2 = -log(x[1]+x[2]) + 1/(2*(x[1]+x[2])) + log(x[2]) - 1/(2*x[2]) - integrate(func, 0, 1, alpha=parms[1], beta=parms[2])$value)
}

ss3 <- multiroot(f = model2, parms=c(10, 10), start = c(1, 1))
ss4 = multiroot(f=model2, parms=c(10, 10/0.01), start=c(0.01,0.01))


ps = exp(-0.16*rgamma(10000, shape=10, rate=10))
reals = sort(rbinom(10000, size=1000, prob=ps))
approx = sort(rbetabinom.ab(10000, size=1000, shape1=ss2$root[1], shape2=ss2$root[2]))
data = data.frame(real=reals, approx=approx)
ggplot(data=data, aes(x=real, y=approx)) + geom_point() +
  geom_abline(intercept=0, slope=1, colour="red") + 
  xlab("True distribution quantiles") + ylab("Approximation distribution quantiles")
ggsave("/home/raphael/dissertation/figures/qqBlow.pdf", width=8, height=6)
qqplot(reals, approx)
abline(0, 1, col="red")
