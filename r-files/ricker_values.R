library(rootSolve)
library(ggplot2)
ata = read.csv("/home/raphael/abc.txt", sep=" ", header=F)
lik = data[nrow(data)-1, 3]

xs = seq(from=0.00, to=10, length.out=10000)
func = function(x, r=17, K=1) return(r*x*exp(-x/K))
ys = c()
for (x in xs) ys = c(ys, func(x) - x)
plot(xs, ys, type="l")
abline(0, 0)

ys2 = c()
for (x in xs) ys2 = c(ys2, func(func(x)) - x)
plot(xs, ys2, type="l")
abline(0, 0)

ys3 = c()
for (x in xs) ys3 = c(ys3, func(func(func(x))) - x)
plot(xs, ys3, type="l")
abline(0, 0)

suite = c(3)
r = 0.5
for (i in 2:100) suite = c(suite, func(suite[i-1], r=r, K=10))
plot(suite, type="l")
abline(log(r), 0)
data = data.frame(xs=1:100, suite=suite)
ggplot(data=data, aes(x=xs, y=suite)) + geom_line() +
  ylab("Population Size") + xlab("Generation")

uniroot.all(function(x) func(func(x))-x, c(0, 20), n=10000000)

bs = seq(from=0, to=7, length.out=100)
uniques = list()
count = 1
for (b in bs) {
  suite = c(2)
  for (i in 2:10000) suite = c(suite, round(func(suite[i-1], r=b), 2))
  uniques[[count]] = unique(suite[950:1000])
  count = count + 1
}
data = data.frame(x=bs, )


logistic.map <- function(r, x, N, M){
  z <- 1:N
  z[1] <- x
  for(i in c(1:(N-1))){
    z[i+1] <- r *z[i]  * exp(-z[i])
  }
  z[c((N-M):N)]
}

my.r <- seq(0, 20, by=0.003)
Orbit <- sapply(my.r, logistic.map,  x=0.1, N=1000, M=50)

Orbit <- as.vector(Orbit)
r <- sort(rep(my.r, 51))

plot(Orbit ~ r, pch=".")

data = data.frame(x=r, y=Orbit)
ggplot(data=data, aes(x=x, y=y)) + geom_point(size=0.1) + 
  scale_x_continuous(breaks=c(1, exp(2), 2*exp(2)), labels=c("1", "e²", "2e²")) +
  theme(axis.text.x=element_text(face="bold", size=16), axis.title=element_text(size=16))+
  ylab("Values in the orbit") + xlab("Value of bifurcation parameter r")

suite = c(7)
for (i in 2:50) suite = c(suite, func(suite[i-1], r=exp(2.5)))
suite2= c(7.1)
for (i in 2:50) suite2 = c(suite2, func(suite2[i-1], r=exp(2.5)))

data = data.frame(x=1:50, y=suite)
data2 = data.frame(x=1:50, y=suite2)

ggplot(data=data, aes(x=x, y=y)) + geom_line() +
  geom_line(data=data2, aes(x=x, y=y), colour="red") +
  ylab("Population Size") + xlab("Generations")

suite = c(7)
for (i in 2:50) suite = c(suite, func(suite[i-1], r=exp(0.3), K=922))
suite2= c(7.1)
for (i in 2:50) suite2 = c(suite2, func(suite2[i-1], r=exp(0.3), K=922))

data = data.frame(x=1:50, y=suite)
data2 = data.frame(x=1:50, y=suite2)

ggplot(data=data, aes(x=x, y=y)) + geom_line() +
  geom_line(data=data2, aes(x=x, y=y), colour="red") +
  ylab("Population Size") + xlab("Generations")

rs = seq(from=exp(2), to=exp(4), length.out=1000)
ns = c(7)
ys = c(rpois(1, lambda=10*ns[1]))
for (i in 2:50) {
  ns = c(ns, func(ns[i-1], r=exp(3.8)))
  ys = c(ys, rpois(1, lambda=10*ns[i]))
}
loglik = c()
for (r in rs) {
  lik = function(y, n, phi=10) return(-phi*n + y*log(phi*n) - lgamma(y+1))  
  loglik = c(loglik, sum(lik(ys, ns)))
}
plot(log(rs), loglik, type="l")


func2 = function(x, r=17, K=1, sigma=1) return(r*x*exp(-x/K)*rlnorm(1, 0, sigma))
suite = c(200)
obs = c(rpois(1, 5*200))
for (i in 2:77) {
  suite = c(suite, func2(suite[i-1], r=1.48, K=577, sigma=0.6))
  obs = c(obs, rpois(1, 5*suite[i]))
}
plot(obs, type="l")
plot(suite, type="l")
suite2 = c(200)
for (i in 2:77) {
  suite2 = c(suite2, func2(suite2[i-1], r=1.48, K=577, sigma=0.6))
}
lines(suite2, col="red")

