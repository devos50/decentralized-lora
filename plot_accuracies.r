library(ggplot2)

dat <- read.csv("data/accuracies_n64_wikitext.csv")

ggplot(dat, aes(x=round, y=loss, group=group, color=group)) +
  geom_line() +
  xlab("Round") +
  ylab("Loss") +
  theme_bw() +
  theme(legend.position=c(0.8, 0.55), legend.background = element_rect(color = "black", fill = "white", size = 0.3)) +
  labs(color="Algorithm")

ggsave("plots/loss_n64_wikitext.pdf", width=4.3, height=2.3)
