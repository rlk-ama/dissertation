params = c("p", "n0", "sigmap", "delta", "sigmad")
proposal = "optimal"
particles = 200
tol = 0

for (param in params) {
  data = read.table(sprintf("/home/raphael/mle_%s_%s_%s_blowfly.txt",
                            param, proposal, particles, tol), header=F)
  end = nrow(data)-1
  dataFrame = data.frame(x=data[1:end, 1], y=data[1:end, 2])
  ggplot(data=dataFrame, aes(x=x, y=y)) + geom_line() + 
    geom_vline(xintercept=data[end+1, 1], colour="red", linetype="dashed") +
    geom_vline(xintercept=data[end+1, 2], linetype="dashed") +
    ylab("Loglikelihood") + xlab("Parameter")
  ggsave(sprintf("/home/raphael/dissertation/figures/mleBlowfly%s.pdf", param), width=8, height=6)
}


optimal.50.5 = as.numeric(
  read.csv("/home/raphael/stability-blowfly/mle_sigmap_50_optimal_5.txt", sep=" ", header=F))
prior.50.5 = as.numeric(
  read.csv("/home/raphael/stability-blowfly/mle_sigmap_50_prior_5.txt", sep=" ", header=F))
optimal.50.0 = as.numeric(
  read.csv("/home/raphael/stability-blowfly/mle_n0_50_optimal_0.txt", sep=" ", header=F))
prior.50.0 = as.numeric(
  read.csv("/home/raphael/stability-blowfly/mle_sigmap_50_prior_0.txt", sep=" ", header=F))
optimal.100.5 = as.numeric(
  read.csv("/home/raphael/stability-blowfly/mle_n0_100_optimal_5.txt", sep=" ", header=F))
prior.100.5 = as.numeric(
  read.csv("/home/raphael/stability-blowfly/mle_n0_100_prior_5.txt", sep=" ", header=F))
optimal.100.0 = as.numeric(
  read.csv("/home/raphael/stability-blowfly/mle_n0_100_optimal_0.txt", sep=" ", header=F))
prior.100.0 = as.numeric(
  read.csv("/home/raphael/stability-blowfly/mle_n0_100_prior_0.txt", sep=" ", header=F))



sd(optimal.50.5)
sd(optimal.100.5) #decreases with inner
mean((optimal.50.5-40)^2)
mean((optimal.100.5-40)^2) #decreases with inner

var(optimal.50.0)
var(optimal.100.0) #same
mean((optimal.50.0-40)^2)
mean((optimal.100.0-40)^2) #decreases with inner

var(prior.50.5)
var(prior.100.5) #decreases with inner
mean((prior.50.5-40)^2)
mean((prior.100.5-40)^2) #increases with inner !

var(prior.50.0)
var(prior.100.0) #decreases with inner
mean((prior.50.0-40)^2)
mean((prior.100.0-40)^2) #decreases with inner

var(optimal.50.5)
var(optimal.50.0) #0 varies more
mean((optimal.50.5 - 40)^2)
mean((optimal.50.0-40)^2) #decreases with tol 0


var(prior.50.5)
var(prior.50.0) #0 varies a lot more
mean((prior.50.5 - 40)^2)
mean((prior.50.0-40)^2) #decreases with tol 0

var(optimal.50.5)
var(prior.50.5) #kiff kiff
mean((optimal.50.5 - 40)^2)
mean((prior.50.5-40)^2)  #decreases with prior


var(optimal.50.0)
var(prior.50.0) #prior varies a lot more
mean((optimal.50.0 - 40)^2)
mean((prior.50.0-40)^2) #decreases with optimal


data = as.numeric(read.csv("/home/raphael/stability_likelihood_blowfly_200_optimal_0_100.txt", sep=" ", header=F))
sd(data[data > -Inf])
