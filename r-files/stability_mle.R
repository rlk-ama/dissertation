msevar = function(l, path, true, variable) {
  output = matrix(nrow=length(l), ncol=5)
  colnames(output) = c("Particles", "Var prior", "Var gamma", "MSE prior", "MSE gamma")
  count = 1
  for (number in l) {
    prior = as.numeric(read.csv(paste(path, sprintf("prior_%s_%s.txt", variable, number), sep=""), sep=" ", header=F))
    gamma = as.numeric(read.csv(paste(path, sprintf("gamma_%s_%s.txt", variable, number), sep=""), sep=" ", header=F))
    output[count, ] = c(number, var(prior), var(gamma),
                    mean((prior-true)^2), mean((gamma-true)^2))
    count = count + 1
  }
  return(output)
}


plotmle = function(particles, save.path, path, true, variable) {
  for (i in 1:length(particles)) {
    prior = as.numeric(read.csv(paste(path, sprintf("prior_%s_%s.txt", variable, particles[i]), sep=""), sep=" ", header=F))
    gamma = as.numeric(read.csv(paste(path, sprintf("gamma_%s_%s.txt", variable, particles[i]), sep=""), sep=" ", header=F))
    pdf(paste(save.path, sprintf("_%s_prior_%s.pdf",  variable, particles[i]), sep=""), width=8, height=6)
    plot(prior, type="l", ylim=c(true*2/3, 1.5*true))
    abline(true, 0, col="red")
    dev.off()
    pdf(paste(save.path, sprintf("_%s_gamma_%s.pdf",  variable, particles[i]), sep=""), width=8, height=6)
    plot(gamma, type="l", ylim=c(true*2/3, 1.5*true))
    abline(true, 0, col="red")
    dev.off()
    
  }
}


PATH = "/home/raphael/mle4/"
SAVE_PATH = "/home/raphael/dissertation/figures/mle2"
true_r = 44.7
true_phi = 10
true_sigma = 0.3
PARTICLES = c(50, 100, 200, 500, 1000, 1500)


output_r = msevar(PARTICLES, PATH, true_r, 'r')
output_phi = msevar(PARTICLES, PATH, true_phi, 'phi')
output_sigma = msevar(PARTICLES, PATH, true_sigma, 'sigma')
plotmle(PARTICLES, SAVE_PATH, PATH, true_r, 'r')
plotmle(PARTICLES, SAVE_PATH, PATH, true_phi, 'phi')
plotmle(PARTICLES, SAVE_PATH, PATH, true_sigma, 'sigma')
