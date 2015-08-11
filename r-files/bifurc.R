library(ggplot2)
xs = seq(from=0, to=exp(2)+1, length.out=1000)
eq1 = rep(0, 1000)
eq1.bold = eq1[xs < 1]
eq1.dashed = eq1[xs > 1]
eq2 = log(xs)
eq2.bold = eq2[xs > 1 & xs < exp(2)]
eq2.dashed1 = eq2[xs < 1]
eq2.dashed2 = eq2[ xs > exp(2)]
dataset1 = data.frame(xs=xs[xs < 1], eq1.bold=eq1.bold)
dataset2 = data.frame(xs=xs[xs > 1], eq1.dashed=eq1.dashed)
dataset3 = data.frame(xs=xs[xs > 1 & xs < exp(2)], eq2.bold=eq2.bold)
dataset4 = data.frame(xs=xs[xs < 1], eq2.dashed1=eq2.dashed1)
dataset5 = data.frame(xs=xs[xs > exp(2)], eq2.dashed2=eq2.dashed2)
ggplot(data=dataset1, aes(x=xs, y=eq1.bold)) + geom_line() +
  geom_line(data=dataset2, aes(x=xs, y=eq1.dashed), linetype="dashed") +
  geom_line(data=dataset3, aes(x=xs, y=eq2.bold)) +
  geom_line(data=dataset4, aes(x=xs, y=eq2.dashed1), linetype="dashed") +
  geom_line(data=dataset5, aes(x=xs, y=eq2.dashed2), linetype="dashed") + 
  ylim(-1, 3) + ylab("Equilibrium lines") + xlab("Value of parameter r") + 
  scale_x_continuous(breaks=c(1, exp(2)), labels=c("1", "e²")) + 
  theme(axis.text.x=element_text(face="bold", size=16), axis.title=element_text(size=16))

