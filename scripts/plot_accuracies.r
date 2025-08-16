library(ggplot2)

for (dataset in c("newsgroups", "emotion", "wikitext")) {
  dat <- read.csv(paste0("../data/accuracies_n64_", dataset, ".csv"))

  ggplot(dat, aes(x=round, y=accuracy, group=group, color=group)) +
    geom_line() +
    xlab("Round") +
    ylab("Accuracy") +
    theme_bw() +
    theme(legend.position=c(0.8, 0.45), legend.background = element_rect(color = "black", fill = "white", size = 0.3)) +
    labs(color="Algorithm")

  ggsave(paste0("../plots/accuracies_n64_", dataset, ".pdf"), width=4.3, height=2.3)

  # Loss
  ggplot(dat, aes(x=round, y=loss, group=group, color=group)) +
    geom_line() +
    xlab("Round") +
    ylab("Loss") +
    theme_bw() +
    theme(legend.position=c(0.8, 0.45), legend.background = element_rect(color = "black", fill = "white", size = 0.3)) +
    labs(color="Algorithm")

  ggsave(paste0("../plots/loss_n64_", dataset, ".pdf"), width=4.3, height=2.3)
}
