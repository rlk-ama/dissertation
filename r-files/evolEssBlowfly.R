library(ggplot2)
library(reshape2)
repetitions = c(50, 100, 500, 1000, 10000)
data = read.table(sprintf("/home/raphael/abc_%s_500.txt", 100), header=F)
dataObs = data.frame(x=1:85, "Observations"=data[, 1])
dataFrame = data.frame(x=1:85)
dataFrame[paste(50)] = data[, 2]
for (repe in repetitions) {
  data = read.table(sprintf("/home/raphael/abc_%s_500.txt", repe), header=F)
  dataFrame[paste(repe)] = data[, 2]
}

dataFrameLong = melt(dataFrame, id.vars="x")
ggplot(data=dataObs, aes(x=x, y=Observations)) + geom_line(colour="red") + 
  geom_line(data=dataFrameLong, aes(x=x, y=value, colour=variable)) + 
  scale_colour_discrete(name = "Number of inner samples")  +
  ylab("Effective sample size") + xlab("Time step")


particles = c(100, 500, 1000, 10000)
data = read.table(sprintf("/home/raphael/abc_%s_%s.txt", 200, 100), header=F)
for (part in particles) {
  data = read.table(sprintf("/home/raphael/abc_%s_%s.txt", 200, part), header=F)
  dataFrame = data.frame(x=1:85, y=data[, 2])
  print(ggplot(data=dataFrame, aes(x=x, y=y)) + geom_line())
}

dataFrameLong = melt(dataFrame, id.vars="x")
 + 
  geom_line(data=dataFrameLong, aes(x=x, y=value, colour=variable)) + 
  scale_colour_discrete(name = "Number of inner samples")  +
  ylab("Effective sample size") + xlab("Time step")
