# R reference run: generates reference outputs for Python port validation
suppressWarnings(suppressMessages({
  library(zoo)
  library(quantreg)
  library(MASS)
}))
e <- new.env()
RDIR <- if (dir.exists("R")) "R" else "target_repo/R"
DDIR <- if (dir.exists("data")) "data" else "target_repo/data"
for (f in list.files(DDIR, full.names=TRUE)) load(f, envir=e)

# stub for progress_bar
progress_bar <- list(new = function(...) list(tick = function(...) invisible(NULL)))
assign("progress_bar", progress_bar, envir = .GlobalEnv)
# progress:: stub
assign("progress", list(progress_bar = progress_bar), envir = .GlobalEnv)

src <- c("VAR.R","QVAR.R","IRF.R","FEVD.R","Wold.R","ConnectednessTable.R","TimeConnectedness.R",
         "FrequencyConnectedness.R","JointConnectedness.R","ExtendedJointConnectedness.R",
         "TVPVAR.R","BayesPrior.R","MinnesotaPrior.R","UninformativePrior.R","ElasticNetVAR.R")
for (s in src) source(file.path(RDIR, s), local=TRUE)

dump_num <- function(name, x, file) {
  # dump a numeric object compactly: name, dims, values
  cat("###", name, "\n", file=file, append=TRUE)
  write.table(format(as.numeric(x), digits=12, trim=TRUE, scientific=FALSE), file=file,
              append=TRUE, row.names=FALSE, col.names=FALSE)
}

OUT <- "validation/reference_results.txt"
cat("", file=OUT)

show <- function(name, x) {
  cat("\n###", name, "\n")
  print(x, digits=12)
  cat("\n###", name, "\n", file=OUT, append=TRUE)
  if (is.matrix(x) || is.array(x)) {
    write.table(format(as.numeric(x), digits=12, trim=TRUE, scientific=FALSE), file=OUT, append=TRUE,
                row.names=FALSE, col.names=FALSE)
  } else {
    write.table(format(as.numeric(x), digits=12, trim=TRUE, scientific=FALSE), file=OUT, append=TRUE,
                row.names=FALSE, col.names=FALSE)
  }
}

# ---------- 1. VAR ----------
dy2012 <- e$dy2012
fit <- VAR(dy2012, configuration=list(nlag=1))
show("VAR_B", fit$B)
show("VAR_Q", fit$Q[,,1])
show("VAR_se", fit$se)

# ---------- 2. FEVD time generalized ----------
fevd <- FEVD(Phi=fit$B, Sigma=fit$Q, nfore=10, type="time", generalized=TRUE)$FEVD
show("FEVD_time_gen", fevd)
fevd_o <- FEVD(Phi=fit$B, Sigma=fit$Q, nfore=10, type="time", generalized=FALSE)$FEVD
show("FEVD_time_orth", fevd_o)

# ---------- 3. ConnectednessTable ----------
ct <- ConnectednessTable(fevd)
show("CT_TCI", ct$TCI)
show("CT_cTCI", ct$cTCI)
show("CT_TO", ct$TO)
show("CT_FROM", ct$FROM)
show("CT_NET", ct$NET)
show("CT_NPT", ct$NPT)
show("CT_PCI", ct$PCI)
show("CT_INFLUENCE", ct$INFLUENCE)

# ---------- 4. TimeConnectedness static VAR ----------
dca <- TimeConnectedness(Phi=fit$B, Sigma=fit$Q, nfore=10, generalized=TRUE)
show("TIME_TCI", dca$TCI)

# ---------- 5. QVAR ----------
qfit <- QVAR(dy2012, configuration=list(nlag=1, tau=0.5))
show("QVAR_B", qfit$B)
show("QVAR_Q", qfit$Q[,,1])

# ---------- 6. TVP-VAR ----------
prior <- BayesPrior(dy2012, nlag=1)
show("BAYESPRIOR_bprior", prior$bprior)
tfit <- TVPVAR(dy2012, configuration=list(nlag=1, prior=prior, l=c(0.99,0.99)))
show("TVPVAR_Bt_dim", matrix(dim(tfit$B_t), nrow=1))
show("TVPVAR_Bt_last", tfit$B_t[,,dim(tfit$B_t)[3]])
show("TVPVAR_Qt_last", tfit$Q_t[,,dim(tfit$Q_t)[3]])
show("TVPVAR_Bt_first", tfit$B_t[,,1])
tdca <- TimeConnectedness(Phi=tfit$B_t, Sigma=tfit$Q_t, nfore=10, generalized=TRUE)
show("TVPTIME_TCI_head", tdca$TCI[1:min(5,length(tdca$TCI)),1])
show("TVPTIME_TCI_tail", tdca$TCI[(max(1,length(tdca$TCI)-4)):length(tdca$TCI),1])

# Minnesota prior TVPVAR
mprior <- MinnesotaPrior(0.1, k=4, nlag=1)
mfit <- TVPVAR(dy2012, configuration=list(nlag=1, prior=mprior, l=c(0.99,0.99)))
show("TVPVAR_MINN_Bt_last", mfit$B_t[,,dim(mfit$B_t)[3]])

# ---------- 7. Frequency connectedness ----------
partition <- c(pi+0.00001, pi/4, 0)
fdca <- FrequencyConnectedness(Phi=fit$B, Sigma=fit$Q, nfore=100, partition=partition)
show("FREQ_TCI", fdca$TCI)
show("FREQ_CT_band2", fdca$CT[,,1,2])

# ---------- 8. Joint ----------
jfit <- VAR(e$dy2012, configuration=list(nlag=2))
jdca <- JointConnectedness(Phi=jfit$B, Sigma=jfit$Q, nfore=30)
show("JOINT_TCI", jdca$TCI)
show("JOINT_TO", jdca$TO)
show("JOINT_FROM", jdca$FROM)
show("JOINT_NET", jdca$NET)

# ---------- 9. Extended joint ----------
prior2 <- MinnesotaPrior(0.1, k=4, nlag=1)
bfit <- TVPVAR(e$dy2012, configuration=list(nlag=1, prior=prior2, l=c(0.99,0.99)))
ejdca <- ExtendedJointConnectedness(Phi=bfit$B_t, Sigma=bfit$Q_t, nfore=20)
show("EXTJOINT_TCI", ejdca$TCI)
show("EXTJOINT_NET_last", ejdca$NET[dim(ejdca$NET)[1],])

# ---------- 10. ElasticNet (LASSO) if glmnet available ----------
if (requireNamespace("glmnet", quietly=TRUE)) {
  efit <- ElasticNetVAR(dy2012, configuration=list(nlag=1, alpha=1, nfolds=10, loss="mae"))
  show("LASSO_B", efit$B)
  show("LASSO_Q", efit$Q[,,1])
}

cat("\nDone.\n")
