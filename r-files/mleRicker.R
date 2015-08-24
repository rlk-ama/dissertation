params = c("r", "phi", "sigma")
for (param in params) {
  data = read.table(sprintf("/home/raphael/mle_%s_ricker.txt", param), header=F)
  end = nrow(data)-1
  dataFrame = data.frame(x=data[1:end, 1], y=data[1:end, 2])
  ggplot(data=dataFrame, aes(x=x, y=y)) + geom_line() + 
    geom_vline(xintercept=data[end+1, 1], colour="red", linetype="dashed") +
    geom_vline(xintercept=data[end+1, 2], linetype="dashed") +
    ylab("Loglikelihood") + xlab("Parameter")
  ggsave(sprintf("/home/raphael/dissertation/figures/mleRicker%s.pdf", param), width=8, height=6)
}
