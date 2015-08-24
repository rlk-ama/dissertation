library(ggplot2)
multinom = read.table("/home/raphael/test_multinom.txt", header=F)
ricker = read.table("/home/raphael/timing_prior.txt", header=F)
ricker = read.table("/home/raphael/timing_gamma.txt", header=F)
slope = (ricker[nrow(ricker), 2]-ricker[1, 2])/(ricker[nrow(ricker), 1]-ricker[1, 1])
intercept = ricker[1, 2]
ggplot(data=ricker, aes(x=ricker[, 1], y=ricker[, 2])) + geom_line() +
 geom_abline(intercept=intercept , slope=slope, colour="red") +
  ylab("Average running time") + xlab("Number of particles")
ggsave("/home/raphael/dissertation/figures/runningRicker.pdf", width=8, height=6)

plot(multinom[, 1], multinom[, 2], type="l")

data = read.table("/home/raphael/pickled_ricker.txt", header=F)
data["obs"] = rpois(50, 10*data[, 2])
ggplot(data=data, aes(x=1:50, y=data[, "obs"])) + geom_line() +
  geom_line(data=data, aes(x=1:50, y=data[, 1]), col="red") +
  ylab("Observations") + xlab("Generations")
ggsave("/home/raphael/dissertation/figures/obsEstimRicker.pdf", width=8, height=6)

ggplot(data=data, aes(x=1:50, y=data[, 4])) + geom_line() +
  ylab("ESS") + xlab("Generations") + ylim(0, 1000)
ggsave("/home/raphael/dissertation/figures/ESSRickerGamma.pdf", width=8, height=6)
