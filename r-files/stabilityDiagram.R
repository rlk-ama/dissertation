library(rootSolve)
func = function(x) x*exp(1-1/x)
func2 = function(x) x*exp(1+exp(-x-1)/x)
xs = seq(from=0.05, to=10, length.out=10000)
ys = func2(xs)
plot(xs, ys, type="l")

data = read.table("/home/raphael/abc_50_100_0.05.txt", header=F)
plot(data[, 1], type="l")
lines(data[, 2], col="red")
obs = as.numeric(read.csv("/home/raphael/experiment1_obs.txt", sep=" ", header=F))
lines(obs, col="red")
plot(obs, type="l")

alpha=1/0.21
P = 4.45
N0 = 200
ancestor = 337
p = alpha/(P*ancestor*exp(-ancestor/N0)+alpha)
size = alpha
rnbinom(1, alpha, p)


mu = function(u, x, y) {
  return(u-x*(1+x*exp(u)*(log(y/x)-1)))
}

xs = seq(from=0.01, to=10, length.out=200)
ys = seq(from=1, to=500, length.out=200)
grid = expand.grid(xs, ys)
sols = matrix(ncol=2)
for (i in 1:nrow(grid)) {
  out = tryCatch(uniroot(mu, x=grid[i, 1], y=grid[i, 2], lower=0, upper=100)$root,
                 error=function(cond) return(0))
  if (out==0) {sols = rbind(sols, c(grid[i,1], grid[i, 2]))}
}
plot(log(sols[-1, 1]), log(sols[-1, 2]))
