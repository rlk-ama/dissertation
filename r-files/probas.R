func = function(v, n, x, b, a) {
  coeff = choose(n,x)*b^a/gamma(a)
  inte = exp(-(x+b)*v)*(1-exp(-v))^(n-x)*v^(a-1)
  return(coeff*inte)
}

func2 = function(n, x, b, a) {
  coeff = choose(n, x)*b^a
  summand = c()
  for (k in 0:(n-x)) summand = c(summand, choose(n-x, k)*(-1)^k*1/(x+b+k)^a)
  return(coeff*sum(summand))
}

func2bis = function(n, x, b, a) {
  summand = c()
  for (k in 0:(n-x)) {
    coeff = lgamma(a) + lgamma(n-x+1) - lgamma(k+1) - lgamma(n-x-k+1) - a*log(x+b+k)
    summand = c(summand, (-1)^k*exp(coeff))
  }
  #coeff = b^a*choose(n, x)/gamma(a)
  coeff = b^a/gamma(a)
  return(coeff*sum(summand))
}

vals =c()
n = 45
for (i in 0:n) {
  vals = c(vals, integrate(func, n=n, x=i, b=10, a=10, lower=0, upper=Inf)$value)
}
#plot(0:n, vals, type="l", ylim=c(-0.01, 0))
sum(vals)

vals2 = c()
for (i in 0:n) {
  vals2 = c(vals2, func2(n=n, x=i, b=10, a=10))
}
plot(0:n, vals2-vals, type="l")
sum(vals2)


func3 = function(n, x) return((1-exp(-x))^(n-x))
func4 = function(n, x) {
  summand = c()
  for (k in 0:(n-x)) summand = c(summand, choose(n-x, k)*(-1)^k*exp(-k*x))
  return(sum(summand))
}
test1 = func3(20, 0:20)
test2 = c()
for (i in 0:20) test2 = c(test2, func4(20, i))
plot(test1, type="l")
lines(test2, col="red")


choose(30, 0:30)
1/(25+0:30)^10
choose(30, 0:30)*1/(25+0:30)^10
choose(45, 15)*10^10*sum((-1)^(0:30)*choose(30, 0:30)*1/(25+0:30)^10)

func5 = function(v, n, x, b, a) {
  inte = exp(-(x+b)*v)*(1-exp(-v))^(n-x)*v^(a-1)
  return(inte)
}
func6 = function(n, x, b, a) {
  summand = c()
  for (k in 0:(n-x)) summand = c(summand, choose(n-x, k)*(-1)^k*1/(x+b+k)^a)
  return(gamma(a)*sum(summand))
}

vals = c()
n = 100
for (i in 0:n) {
  vals = c(vals, integrate(func5, n=n, x=i, b=10, a=10, lower=0, upper=Inf)$value)
}


vals2 = c()
for (i in 0:n) {
  vals2 = c(vals2, func6(n=n, x=i, b=10, a=10))
}
plot(0:n, vals2-vals, type="l")
sum(vals2)


func7 = function(v, n, x, b, a) {
  inte = exp(-(x+b)*v + (n-x)*log(1-exp(-v)) + (a-1)*log(v))
  return(inte)
}
func8 = function(n, x, b, a) {
  summand = c()
  for (k in 0:(n-x)) summand = c(summand, (-1)^k*exp(lgamma(a) + lgamma(n-x+1) - lgamma(k+1) - lgamma(n-x-k+1) - a*log(x+b+k)))
  return(sum(summand))
}

vals = c()
n = 60
for (i in 0:n) {
  vals = c(vals, integrate(func5, n=n, x=i, b=10, a=10, lower=0, upper=Inf)$value)
}

vals2 = c()
for (i in 0:n) {
  vals2 = c(vals2, func8(n=n, x=i, b=10, a=10))
}
plot(0:n, vals2-vals, type="l")
sum(vals2)

ys = beta(10:20, 10:20)
plot(ys, type="l")


meanbeta = function(n, a, b) return(n*a/(a+b))
stdbeta = function(n, a, b) return(sqrt(n*a*b*(a+b+n)/((a+b)^2*(a+b+1))))
n = 360
a =  149.9153684462072
b =   16.458142414956917
nexti = 273
meanbeta(n, a, b) - 4*stdbeta(n, a, b)
ys = rbetabinom.ab(1000, size=n, shape1=a, shape2=b)
plot(ys, ylim=c(nexti-20, n))
abline(nexti, 0)

yy = c()
for (i in 1:10000) {
  p = rbeta(1, shape1=a, shape2=b)
  y = rbinom(1, size=n, prob=p)
  yy = c(yy, y)
}
sum(yy <= nexti)
min(yy)
