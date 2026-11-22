suppressWarnings(suppressMessages({library(zoo); library(MASS)}))
e <- new.env()
load("target_repo/data/dy2012.rda", envir=e)
for (s in c("VAR.R","IRF.R","FEVD.R","BayesPrior.R","ConnectednessTable.R","TimeConnectedness.R"))
  source(file.path("target_repo/R", s), local=TRUE)

OUT <- "validation/freq_debug.txt"
cat("", file=OUT)
dumpv <- function(name, x) {
  cat("###", name, "\n", file=OUT, append=TRUE)
  write.table(format(as.numeric(x), digits=12, trim=TRUE, scientific=FALSE),
              file=OUT, append=TRUE, row.names=FALSE, col.names=FALSE)
}

dy2012 <- e$dy2012
fit <- VAR(dy2012, configuration=list(nlag=1))

# 1. BayesPrior Q (hbar) ni tekshirish
prior <- BayesPrior(dy2012, nlag=1)
dumpv("HBAR", prior$Q)
dumpv("VARQ", fit$Q[,,1])

# 2. TVP-TCI[1] — FEVD(OLS B, hbar) bilan
tci_hbar <- TimeConnectedness(Phi=fit$B, Sigma=prior$Q, nfore=10, generalized=TRUE)
dumpv("TCI_HBAR", tci_hbar$TCI)

# 3. Chastota FEVD oraliq qiymatlari (partition = c(pi+1e-5, pi/4, 0), nfore=100)
library(frequencyConnectedness)
partition <- c(pi+0.00001, pi/4, 0)
new_p <- frequencyConnectedness::getPartition(partition, 100)
dumpv("PART1", new_p[[1]])
dumpv("PART2", new_p[[2]])
rng <- sort(unique(do.call(c, new_p)))
dumpv("RNG", rng)

irf <- IRF(Phi=fit$B, Sigma=fit$Q, nfore=100, orth=FALSE)$irf
fftir <- lapply(irf, function(i) apply(i, 2, fft))
fftir <- lapply(1:(100+1), function(j) sapply(fftir, function(i) i[j,]))
denom <- diag(Re(Reduce("+", lapply(fftir, function(i) i %*% fit$Q[,,1] %*% t(Conj(i))/100)[rng])))
dumpv("FREQ_DENOM", denom)
enum_all <- lapply(fftir, function(i) (abs(i %*% fit$Q[,,1]))^2/(100+1))
tab_all <- lapply(enum_all, function(i) sapply(1:nrow(i), function(j) i[j,]/(denom[j]*diag(fit$Q[,,1]))))
dumpv("FREQ_TAB_w1", tab_all[[1]])
dumpv("FREQ_TAB_w51", tab_all[[51]])
tot <- apply(Reduce("+", tab_all[rng]), 2, sum)
dumpv("FREQ_TOT", tot)
dumpv("FREQ_FEVD_w1", t(tab_all[[1]])/tot)
dumpv("FREQ_FEVD_w51", t(tab_all[[51]])/tot)
cat("done\n")
