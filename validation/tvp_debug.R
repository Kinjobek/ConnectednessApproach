e <- new.env()
load("target_repo/data/dy2012.rda", envir=e)
source("target_repo/R/BayesPrior.R")
source("target_repo/R/MinnesotaPrior.R")
source("target_repo/R/UninformativePrior.R")

OUT <- "validation/tvp_debug.txt"
cat("", file=OUT)

dumpv <- function(name, x) {
  cat("###", name, "\n", file=OUT, append=TRUE)
  write.table(format(as.numeric(x), digits=12, trim=TRUE, scientific=FALSE),
              file=OUT, append=TRUE, row.names=FALSE, col.names=FALSE)
}

x <- e$dy2012
prior <- MinnesotaPrior(0.1, k=4, nlag=1)
beta_0.mean <- prior$bprior
beta_0.var <- prior$Vprior
Q <- prior$Q
if (is.null(Q)) Q <- cov(x)
nlag <- 1

create_RHS_NI = function(templag, r, nlag, t){
    K = nlag*(r^2)
    x_t = matrix(0, (t-nlag)*r, K)
    for (i in 1:(t-nlag)){
      ztemp = NULL
      for (j in 1:nlag) {
        xtemp = templag[i,((j-1)*r+1):(j*r)]
        if (i <= 2) cat("DEBUG i=", i, ": xtemp class=", paste(class(xtemp), collapse=","), " dim=", paste(dim(xtemp), collapse="x"), "\n", sep="")
        xtemp = t(kronecker(diag(r),xtemp))
        if (i <= 2) cat("DEBUG i=", i, ": kron+T dim=", paste(dim(xtemp), collapse="x"), "\n", sep="")
        ztemp = cbind(ztemp, xtemp)
        if (i <= 2) cat("DEBUG i=", i, ": ztemp dim=", paste(dim(ztemp), collapse="x"), "\n", sep="")
      }
      x_t[((i-1)*r+1):(i*r),] = ztemp
    }
    return=list(x_t=x_t, K=K)
}

x <- scale(x,TRUE,FALSE)
YX = cbind(x,x)
r = p = n = ncol(x)
m = nlag*(r^2)
k = nlag*r
t = nrow(x)
q = n + p

beta_0_prmean = beta_0.mean
beta_0_prvar = beta_0.var

beta_pred = matrix(0,m,t)
beta_update = matrix(0,m,t)
Rb_t = array(0,c(m,m,t))
Sb_t = array(0,c(m,m,t))
beta_t = array(0, c(k,k,t))
Q_t = array(0, c(r,r,t))

l_2 = 0.99; l_4 = 0.99

yy = x[(nlag+1):t,]
xx = embed(x, nlag+1)[,-c(1:r)]
templag = embed(x, nlag+1)[,-c(1:r)]
RHS1 = create_RHS_NI(templag,r,nlag,t)
Flagtemp = RHS1$x_t
Flag = rbind(matrix(0, k,m), Flagtemp)

dumpv("FLAG_block5", Flag[((5-1)*r+1):(5*r),])

# Kalman filter — faqat birinchi 30 qadam, beta_update va Q_t yozib olamiz
for (irep in 1:30) {
    if (irep==1) {
      Q_t[,,irep] = Q
    } else if (irep > 1) {
      if (irep <= (nlag+1)) {
        Gf_t = 0.1*(t(matrix(x[irep,],nrow=1))%*%(x[irep,]))
      } else {
        Gf_t = t(yy[(irep-nlag),]-xx[(irep-nlag),]%*%t(B[1:r,1:k])) %*% (yy[(irep-nlag),]-xx[(irep-nlag),]%*%t(B[1:r,1:k]))
      }
      Q_t[,,irep] = l_2*Q_t[,,(irep-1)] + (1-l_2)*Gf_t[1:r,1:r]
    }
    if (irep <= (nlag+1)) {
      beta_pred[,irep] = beta_0_prmean
      beta_update[,irep] = beta_pred[,irep]
      Rb_t[,,irep] = beta_0_prvar
    } else if (irep > (nlag+1)) {
      beta_pred[,irep] = beta_update[,(irep-1)]
      Rb_t[,,irep] = (1/l_4)*Sb_t[,,(irep-1)]
    }
    if (irep >= (nlag+1)) {
      Rx = Rb_t[,,irep]%*%t(Flag[((irep-1)*r+1):(irep*r),])
      KV_b = Q_t[,,irep] + Flag[((irep-1)*r+1):(irep*r),]%*%Rx
      KG = Rx%*%MASS::ginv(KV_b)
      beta_update[,irep] = matrix(beta_pred[,irep], ncol=1) + (KG%*%(t(matrix(x[irep,],nrow=1))-Flag[((irep-1)*r+1):(irep*r),]%*%matrix(beta_pred[,irep], ncol=1)) )
      Sb_t[,,irep] = Rb_t[,,irep] - KG%*%(Flag[((irep-1)*r+1):(irep*r),]%*%Rb_t[,,irep])
    }
    bb = matrix(beta_update[,irep], ncol=1)
    splace = 0
    biga = matrix(0, r,r*nlag)
    for (ii in 1:nlag) {
      for (iii in 1:r) {
        biga[iii,((ii-1)*r+1):(ii*r)] = t(bb[(splace+1):((splace+r)),1])
        splace = splace + r
      }
    }
    B = rbind(biga, cbind(diag(r*(nlag-1)), matrix(0, nrow=r*(nlag-1), ncol=r)))
    if ((max(abs(eigen(B)$values))<=1)||(irep==1)){
      beta_t[,,irep] = B
    } else {
      beta_t[,,irep] = beta_t[,,(irep-1)]
      beta_update[,irep] = 0.99*beta_update[,(irep-1)]
    }
}
dumpv("BETA_UPDATE_2", beta_update[,2])
dumpv("BETA_UPDATE_5", beta_update[,5])
dumpv("BETA_UPDATE_10", beta_update[,10])
dumpv("BETA_UPDATE_20", beta_update[,20])
dumpv("BETA_UPDATE_30", beta_update[,30])
dumpv("QT_2", Q_t[,,2])
dumpv("QT_5", Q_t[,,5])
dumpv("QT_30", Q_t[,,30])
dumpv("BIGA_30", beta_t[1:r,1:(r*nlag),30])
cat("done\n")
