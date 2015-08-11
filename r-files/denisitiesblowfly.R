library(VGAM)
dens1 = function(x, obs) {
  coeff = choose(obs, x)*10^10
  summation = sum(choose(obs-x, 0:(obs-x))*(-1)^(0:(obs-x))*1/(0.16*(x+(0:(obs-x))+10)^10))
  return(coeff*summation)
}

normalization = function(obs) {
  partial = 0
  for (i in 0:obs) {
    partial = partial + dens1(i, obs)
  }
  return(partial)
}

normalization2 = function(obs, a, b) {
  partial = 0
  for (i in 0:obs) {
    partial = partial + dens2(i, obs, a, b)
  }
  return(partial)
}

obs=697
xs = 0:obs
ys = c()
for (ob in xs) ys = c(ys, dens1(ob, obs))
ys = ys/normalization(obs)
plot(ys, type="l")
ys.beta = dbetabinom.ab(xs, size=697, shape1=58.23357, shape2=10.01770)
plot(ys.beta, type="l")



dens2 = function(x, n, b, a) {
  coeff = choose(n, x)*b^a
  summ = 0
  for (k in 0:(n-x)) summ = summ + (-1)^k*choose(n-x, k)*1/(x+b+k)^a
  return(coeff*summ)
}
ys.2 = c()
for (x in xs) ys.2 = c(ys.2, dens2(x, obs, 10, 10))
ys.2 = ys.2/normalization2(obs, 10, 10)
plot(ys.2, type="l")


dens3 = function(x, obs, alpha, delta) {
  coeff = choose(obs, x)*alpha^alpha/gamma(alpha)
  inte = function(y) return(exp(-(delta*x+alpha)*y)*(1-exp(-delta*y))^(obs-x)*y^(alpha-1))
  return(coeff*integrate(inte, 0, Inf)$value)
}
ys.int = c()
for (x in xs) ys.int = c(ys.int, dens3(x, obs, 10, 0.16))
plot(ys.int, type="l")
