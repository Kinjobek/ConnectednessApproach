suppressWarnings(suppressMessages(library(zoo)))
e <- new.env()
load("target_repo/data/dy2012.rda", envir=e)
for (s in c("VAR.R","IRF.R","FEVD.R")) source(file.path("target_repo/R", s), local=TRUE)

OUT <- "validation/fevd_debug.txt"
cat("", file=OUT)
dumpv <- function(name, x) {
  cat("###", name, "\n", file=OUT, append=TRUE)
  write.table(format(as.numeric(x), digits=12, trim=TRUE, scientific=FALSE),
              file=OUT, append=TRUE, row.names=FALSE, col.names=FALSE)
}

dy2012 <- e$dy2012
fit <- VAR(dy2012, configuration=list(nlag=1))
Phi <- fit$B; Sigma <- fit$Q[,,1]

irf <- IRF(Phi=Phi, Sigma=Sigma, nfore=10, orth=FALSE)$irf
dumpv("IRF_1", irf[[1]])   # (nfore+1) x k
dumpv("IRF_2", irf[[2]])

PhiL <- lapply(1:10, function(j) sapply(irf, function(i) i[j,]))
dumpv("MH_1", PhiL[[1]])  # k x k
dumpv("MH_2", PhiL[[2]])
denom <- diag(Reduce("+", lapply(PhiL, function(i) i %*% Sigma %*% t(i))))
dumpv("DENOM", denom)
enum <- Reduce("+", lapply(PhiL, function(i) (i %*% Sigma)^2))
dumpv("ENUM", enum)
tab <- sapply(1:nrow(enum), function(j) enum[j,]/(denom[j] * diag(Sigma)))
dumpv("TAB", tab)
fevd <- t(apply(tab, 2, function(i) i/sum(i)))
dumpv("FEVD", fevd)
cat("done\n")
