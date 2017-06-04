#https://www.svm-tutorial.com/2014/10/support-vector-regression-r/
#the objective of the SVM is to find  the optimal separating hyperplane which maximizes the margin of the training data
#https://www.r-bloggers.com/learning-kernels-svm/
#http://www.kdnuggets.com/2017/03/building-regression-models-support-vector-regression.html
##############
#16-1-1.R
##############
library(dmr.claseval)
library(dmr.linclas)
library(dmr.regeval)
library(dmr.trans)
library(dmr.util)

library(lattice)
library(quadprog)
library(kernlab)
library(Matrix)

data(PimaIndiansDiabetes, package="mlbench")
data(BostonHousing, package="mlbench")

set.seed(12)

rpid <- runif(nrow(PimaIndiansDiabetes))
pid.train <- PimaIndiansDiabetes[rpid>=0.33,]
pid.test <- PimaIndiansDiabetes[rpid<0.33,]

rbh <- runif(nrow(BostonHousing))
bh.train <- BostonHousing[rbh>=0.33,-4]
bh.test <- BostonHousing[rbh<0.33,-4]

pid.stdm <- std.all(diabetes~., pid.train)
pid.std.train <- predict.std(pid.stdm, pid.train)
pid.std.test <- predict.std(pid.stdm, pid.test)

bh.stdm <- std.all(medv~., bh.train)
bh.std.train <- predict.std(bh.stdm, bh.train)
bh.std.test <- predict.std(bh.stdm, bh.test)

set.seed(12)

# dataset for plots
kmf.plot <- function(a1, a2) { 2*a1-3*a2+4 }
kmdat.plot <- `names<-`(expand.grid(seq(1, 5, 0.05), seq(1, 5, 0.05)), c("a1", "a2"))
kmdat.plot$f <- kmf.plot(kmdat.plot$a1, kmdat.plot$a2)
kmdat.plot$c <- as.factor(ustep(kmdat.plot$f))

# datasets for parameter estimation examples
kmg <- function(a1, a2, a3, a4) { a1^2+2*a2^2-a3^2-2*a4^2+2*a1-3*a2+2*a3-3*a4+1 }
kmf <- function(a1, a2, a3, a4) { 3*a1+4*a2-2*a3+2*a4-3 }
kmdat <- data.frame(a1=runif(400, min=1, max=5), a2=runif(400, min=1, max=5),
                    a3=runif(400, min=1, max=5), a4=runif(400, min=1, max=5))
kmdat$g <- kmg(kmdat$a1, kmdat$a2, kmdat$a3, kmdat$a4)
kmdat$c <- as.factor(ustep(kmdat$g))
kmdat$f <- kmf(kmdat$a1, kmdat$a2, kmdat$a3, kmdat$a4)

kmdat.train <- kmdat[1:200,]
kmdat.test <- kmdat[201:400,]

# linearly separable training and test subsets
kmdat.ls <- linsep.sub(c~a1+a2+a3+a4, kmdat)
kmdat.train.ls <- kmdat[1:200,][kmdat.ls[1:200],]
kmdat.test.ls <- kmdat[201:400,][kmdat.ls[201:400],]
##############
#16-2-1.R
##############
## functional margin of w with respect to instances from data
## using the cvec vector of {-1, 1} class labels
fmarg <- function(w, data, cvec)
{ cvec*predict.par(list(repf=repf.linear, w=w), data) }

## geometric margin of w with respect to instances from data
## using the cvec vector of {-1, 1} class labels
gmarg <- function(w, data, cvec) { fmarg(w, data, cvec)/l2norm(w[-length(w)]) }

## plot separating and b-margin lines for linear threshold classification
## with 2 attributes
plot.margin <- function(w, data, cvec, b=1, add=FALSE,
                        col.sep="black", col.pos="grey70", col.neg="grey30", ...)
{
  # y value corresponding to x on the regression line represented by w
  lry <- function(x, w) {sum(-w[c(1,3)]/w[2]*c(x, 1)) }
  
  if (!add)
  {
    plot(data[,1][cvec==1], data[,2][cvec==1], col=col.pos,
         xlab="a1", ylab="a2", xlim=range(data[,1]), ylim=range(data[,2]), ...)
    points(data[,1][cvec!=1], data[,2][cvec!=1], col=col.neg, ...)
  }
  
  lines(range(data[,1]), c(lry(min(data[,1]), w),
                           lry(max(data[,1]), w)), col=col.sep, ...)
  lines(range(data[,1]), c(lry(min(data[,1]), w-c(0, 0, b)),
                           lry(max(data[,1]), w-c(0, 0, b))), col=col.pos, ...)
  lines(range(data[,1]), c(lry(min(data[,1]), w+c(0, 0, b)),
                           lry(max(data[,1]), w+c(0, 0, b))), col=col.neg, ...)
  list(fmargin=min(fmarg(w, data, cvec)), gmargin=min(gmarg(w, data, cvec)))
}

# dataset for margin illustration (skip near-boundary instances from kmdat.plot)
kmdat.m <- kmdat.plot[abs(kmdat.plot$f)>2,c("a1", "a2", "c")]
kmdat.m <- kmdat.m[sample(nrow(kmdat.m), 100),]

# parameter vector for margin demonstration
w.m <- c(1, -2)
# predictions with intercept 0
p0.m <- predict.par(list(repf=repf.linear, w=c(w.m, 0)), kmdat.m[,1:2])
# symmetric-margin intercept
w.m <- c(w.m, -(max(p0.m[kmdat.m$c==0])+min(p0.m[kmdat.m$c==1]))/2)

# minimum functional margin
min(fmarg(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1))
# minimum geometric
min(gmarg(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1))

# scale parameters to get minimum functional margin of 1
w.m <- w.m/min(fmarg(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1))
# minimum functional margin after parameter scaling (1)
min(fmarg(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1))
# minimum geometric margin after parameter scaling (unchanged)
min(gmarg(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1))

plot.margin(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1)
##############
#16-2-2.R
##############
## linear SVM parameter estimation using primal-form quadratic programming
## solvers: "solve.QP" or "ipop"
svm.linear.prim <- function(formula, data, svthres=1e-9, inf=1e3, solver="solve.QP")
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes
  
  cvec <- 2*as.num0(data[[class]])-1  # class vector using {-1, 1} labels
  amat <- cbind(as.matrix(data[,aind]), intercept=1)  # attribute value matrix
  
  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(rbind(cbind(diag(sum(aind)), 0), 0))$mat,
                 dvec=rep(0, sum(aind)+1),
                 Amat=t(rmm(amat, cvec)),
                 bvec=rep(1, nrow(data)))
  else if (solver=="ipop")
    args <- list(c=rep(0, sum(aind)+1),
                 H=rbind(cbind(diag(sum(aind)), 0), 0),
                 A=rmm(amat, cvec),
                 b=rep(1, nrow(data)),
                 l=rep(-inf, sum(aind)+1),
                 u=rep(inf, sum(aind)+1),
                 r=rep(inf, nrow(data)))
  else stop("Unknown solver: ", solver)
  
  qp <- do.call(solver, args)
  w <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  sv <- unname(which(cvec*predict.par(list(repf=repf.linear, w=w),
                                      data[,aind,drop=FALSE])<=1+svthres))
  list(model=`class<-`(list(repf=repf.threshold(repf.linear), w=w), "par"), sv=sv)
}

# estimate linear SVM model parameters
svm.p.ls <- svm.linear.prim(c~a1+a2+a3+a4, kmdat.train.ls)

# misclassification error
err(predict(svm.p.ls$model, kmdat.train.ls[,1:4]), kmdat.train.ls$c)
err(predict(svm.p.ls$model, kmdat.test.ls[,1:4]), kmdat.test.ls$c)
##############
#16-2-3.R
##############
# hard-margin SVM
svm.mh <- svm.linear.prim(c~., kmdat.m, solver="ipop")

# optimal separating and margin lines
plot.margin(svm.mh$model$w, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1)

# suboptimal separating and margin lines for comparison
plot.margin(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1, add=TRUE, lty=3)
##############
#16-2-4.R
##############
## linear SVM parameter estimation using dual-form quadratic programming
## solvers: "solve.QP" or "ipop"
svm.linear.dual <- function(formula, data, svthres=1e-3, inf=1e3, solver="solve.QP")
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes
  
  cvec <- 2*as.num0(data[[class]])-1  # class vector using {-1, 1} labels
  ccmat <- outer(cvec, cvec)          # class-class product matrix
  amat <- as.matrix(data[,aind])      # attribute value matrix
  dpmat <- amat%*%t(amat)             # dot product matrix
  
  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(dpmat*ccmat)$mat,
                 dvec=rep(1, nrow(data)),
                 Amat=matrix(c(cvec, diag(1, nrow(data))), nrow=nrow(data)),
                 bvec=rep(0, nrow(data)+1),
                 meq=1)
  else if (solver=="ipop")
    args <- list(c=rep(-1, nrow(data)),
                 H=dpmat*ccmat,
                 A=cvec,
                 b=0,
                 l=rep(0, nrow(data)),
                 u=rep(inf, nrow(data)),
                 r=0)
  else
    stop("Unknown solver: ", solver)
  
  qp <- do.call(solver, args)
  alpha <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  sv <- which(alpha>svthres)
  w <- c(colSums(rmm(amat[sv,], cvec[sv]*alpha[sv])))  # no intercept yet
  p0 <- predict.par(list(repf=repf.linear, w=c(w, 0)), data[,aind,drop=FALSE])
  w <- c(w, intercept=-(max(p0[cvec==-1])+min(p0[cvec==1]))/2)
  list(model=`class<-`(list(repf=repf.threshold(repf.linear), w=w), "par"), sv=sv)
}

# estimate linear SVM model parameters
svm.d.ls <- svm.linear.dual(c~a1+a2+a3+a4, kmdat.train.ls)

# misclassification error
err(predict(svm.d.ls$model, kmdat.train.ls[,1:4]), kmdat.train.ls$c)
err(predict(svm.d.ls$model, kmdat.test.ls[,1:4]), kmdat.test.ls$c)
##############
#16-2-5.R
##############
## linear soft-margin SVM parameter estimation using quadratic programming
## solvers: "solve.QP" or "ipop"
svm.linear <- function(formula, data, cost=1, svthres=1e-3, solver="solve.QP")
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes
  
  cvec <- 2*as.num0(data[[class]])-1  # class vector using {-1, 1} labels
  ccmat <- outer(cvec, cvec)          # class-class product matrix
  amat <- as.matrix(data[,aind])      # attribute value matrix
  dpmat <- amat%*%t(amat)             # dot product matrix
  
  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(dpmat*ccmat)$mat,
                 dvec=rep(1, nrow(data)),
                 Amat=matrix(c(cvec, diag(1, nrow(data)), diag(-1, nrow(data))),
                             nrow=nrow(data)),
                 bvec=c(0, rep(0, nrow(data)), rep(-cost, nrow(data))),
                 meq=1)
  else if (solver=="ipop")
    args <- list(c=rep(-1, nrow(data)),
                 H=dpmat*ccmat,
                 A=cvec,
                 b=0,
                 l=rep(0, nrow(data)),
                 u=rep(cost, nrow(data)),
                 r=0)
  else
    stop("Unknown solver: ", solver)
  
  qp <- do.call(solver, args)
  alpha <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  sv <- which(alpha>svthres)
  w <- c(colSums(rmm(amat[sv,], cvec[sv]*alpha[sv])))  # no intercept yet
  i <- which.min(abs(alpha-cost/2))
  w <- c(w, intercept=cvec[i]-unname(predict.par(list(repf=repf.linear, w=c(w, 0)),
                                                 data[i,aind,drop=FALSE])))
  list(model=`class<-`(list(repf=repf.threshold(repf.linear), w=w), "par"), sv=sv)
}

# linear SVM for the artificial data
svm.s <- svm.linear(c~a1+a2+a3+a4, kmdat.train)
svm.s.ls <- svm.linear(c~a1+a2+a3+a4, kmdat.train.ls)

svm.s.01 <- svm.linear(c~a1+a2+a3+a4, kmdat.train, cost=0.1)
svm.s.ls.01 <- svm.linear(c~a1+a2+a3+a4, kmdat.train.ls, cost=0.1)

svm.s.10 <- svm.linear(c~a1+a2+a3+a4, kmdat.train, cost=10)
svm.s.ls.10 <- svm.linear(c~a1+a2+a3+a4, kmdat.train.ls, cost=10)

# linear SVM for the Pima Indians Diabetes data
pid.svm.s <- svm.linear(diabetes~., pid.std.train)
pid.svm.s.01 <- svm.linear(diabetes~., pid.std.train, cost=0.1)
pid.svm.s.10 <- svm.linear(diabetes~., pid.std.train, cost=10)

# training set misclassification error
err(predict(svm.s$model, kmdat.train[,1:4]), kmdat.train$c)
err(predict(svm.s.01$model, kmdat.train[,1:4]), kmdat.train$c)
err(predict(svm.s.10$model, kmdat.train[,1:4]), kmdat.train$c)

err(predict(svm.s.ls$model, kmdat.train.ls[,1:4]), kmdat.train.ls$c)
err(predict(svm.s.ls.01$model, kmdat.train.ls[,1:4]), kmdat.train.ls$c)
err(predict(svm.s.ls.10$model, kmdat.train.ls[,1:4]), kmdat.train.ls$c)

err(factor(predict(pid.svm.s$model, pid.std.train[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.std.train$diabetes)
err(factor(predict(pid.svm.s.01$model, pid.std.train[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.std.train$diabetes)
err(factor(predict(pid.svm.s.10$model, pid.std.train[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.std.train$diabetes)

# test set misclassification error
err(predict(svm.s$model, kmdat.test[,1:4]), kmdat.test$c)
err(predict(svm.s.01$model, kmdat.test[,1:4]), kmdat.test$c)
err(predict(svm.s.10$model, kmdat.test[,1:4]), kmdat.test$c)

err(predict(svm.s.ls$model, kmdat.test.ls[,1:4]), kmdat.test.ls$c)
err(predict(svm.s.ls.01$model, kmdat.test.ls[,1:4]), kmdat.test.ls$c)
err(predict(svm.s.ls.10$model, kmdat.test.ls[,1:4]), kmdat.test.ls$c)

err(factor(predict(pid.svm.s$model, pid.std.test[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.test$diabetes)
err(factor(predict(pid.svm.s.01$model, pid.std.test[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.test$diabetes)
err(factor(predict(pid.svm.s.10$model, pid.std.test[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.test$diabetes)
##############
#16-2-6.R
##############
# soft-margin SVM
svm.ms.1 <- svm.linear(c~., kmdat.m, solver="ipop", cost=1)
w.ms.1 <- svm.ms.1$model$w

svm.ms.01 <- svm.linear(c~., kmdat.m, solver="ipop", cost=0.1)
w.ms.01 <- svm.ms.01$model$w

# soft margin: geometric margin corresponding to functional margin of 1
#1/l2norm(w.ms.1[-length(w.ms.1)])
#1/l2norm(w.ms.01[-length(w.ms.01)])

# separating and margin lines for cost=1
plot.margin(w.ms.1, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1, main="Linearly separable")
# separating and margin lines for cost=0.1
plot.margin(w.ms.01, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1, add=TRUE, lty=3)

# the same for linearly inseparable data

kmdat.m.nls <- kmdat.m
kmdat.m.nls$c <- as.factor(ifelse(runif(nrow(kmdat.m))<0.1,
                                  1-as.numchar(kmdat.m$c), as.numchar(kmdat.m$c)))

svm.ms.nls.1 <- svm.linear(c~., kmdat.m.nls, solver="ipop", cost=1)
w.ms.nls.1 <- svm.ms.nls.1$model$w

svm.ms.nls.01 <- svm.linear(c~., kmdat.m.nls, solver="ipop", cost=0.1)
w.ms.nls.01 <- svm.ms.nls.01$model$w

# soft margin: geometric margin corresponding to functional margin of 1
#1/l2norm(w.ms.nls.1[-length(w.ms.nls.1)])
#1/l2norm(w.ms.nls.01[-length(w.ms.nls.01)])

# separating and margin lines for cost=1
plot.margin(w.ms.nls.1, kmdat.m.nls[,1:2], 2*as.num0(kmdat.m.nls$c)-1,
            main="Linearly inseparable")
# separating and margin lines for cost=0.1
plot.margin(w.ms.nls.01, kmdat.m.nls[,1:2], 2*as.num0(kmdat.m.nls$c)-1,
            add=TRUE, lty=3)
##############
#16-3-1.R
##############
## plot regression tube lines for linear regression
## with a single attributes
plot.tube <- function(w, data, eps, add=FALSE,
                      col.point="black", col.line="black", ...)
{
  # y value corresponding to x on the regression line represented by w
  lry <- function(x, w) {sum(w*c(x, 1)) }
  
  if (!add)
    plot(data[,1], data[,2], col=col.point,
         xlab="a1", ylab="h", xlim=range(data[,1]), ylim=range(data[,2]), ...)
  
  lines(range(data[,1]), c(lry(min(data[,1]), w), lry(max(data[,1]), w)),
        col=col.line)
  lines(range(data[,1]), c(lry(min(data[,1]), w-c(0, eps)),
                           lry(max(data[,1]), w-c(0, eps))), col=col.line, lty=3)
  lines(range(data[,1]), c(lry(min(data[,1]), w+c(0, eps)),
                           lry(max(data[,1]), w+c(0, eps))), col=col.line, lty=3)
}

# dataset for tube demonstration (take instances with similar a2 values)
kmdat.t <- kmdat.plot[abs(kmdat.plot$a2-mean(kmdat.plot$a2))<1,]
kmdat.t <- kmdat.t[sample(nrow(kmdat.t), 100), c("a1", "f")]

# parameter vector for tube demonstration
w.t <- lm(f~a1, kmdat.t)$coef[2:1]

plot.tube(w.t, kmdat.t, eps=1)
##############
#16-3-2.R
##############
## linear SVR parameter estimation using quadratic programming
## solvers: "solve.QP" or "ipop"
svr.linear <- function(formula, data, eps=0.01, cost=1, svthres=1e-3,
                       solver="solve.QP")
{
  f <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes
  
  fvec <- data[[f]]                   # target function vector
  amat <- as.matrix(data[,aind])      # attribute value matrix
  dpmat <- amat%*%t(amat)             # dot product matrix
  
  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(rbind(cbind(dpmat, -dpmat), cbind(-dpmat, dpmat)))$mat,
                 dvec=c(fvec-eps, -fvec-eps),
                 Amat=matrix(c(rep(1, nrow(data)), rep(-1, nrow(data)),
                               diag(1, 2*nrow(data)), diag(-1, 2*nrow(data))),
                             nrow=2*nrow(data)),
                 bvec=c(0, rep(0, 2*nrow(data)), rep(-cost, 2*nrow(data))),
                 meq=1)
  else if (solver=="ipop")
    args <- list(c=c(-fvec+eps, fvec+eps),
                 H=rbind(cbind(dpmat, -dpmat), cbind(-dpmat, dpmat)),
                 A=c(rep(1, nrow(data)), rep(-1, nrow(data))),
                 b=0,
                 l=rep(0, 2*nrow(data)),
                 u=rep(cost, 2*nrow(data)),
                 r=0)
  else
    stop("Unknown solver: ", solver)
  
  qp <- do.call(solver, args)
  alpha <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  beta <- alpha[1:nrow(data)]-alpha[(nrow(data)+1):(2*nrow(data))]
  sv <- which(abs(beta)>svthres)
  w <- c(colSums(rmm(amat[sv,], beta[sv])))  # no intercept yet
  i <- which.min(abs(beta-cost/2))
  w <- c(w, intercept=fvec[i]-unname(predict.par(list(repf=repf.linear, w=c(w, 0)),
                                                 data[i,aind,drop=FALSE]))-
           sign(beta[i])*eps)
  list(model=`class<-`(list(repf=repf.linear, w=w), "par"), sv=sv)
}

# linear SVR for f
svrf <- svr.linear(f~a1+a2+a3+a4, kmdat.train)
svrf.e1 <- svr.linear(f~a1+a2+a3+a4, eps=1, kmdat.train)
svrf.c01 <- svr.linear(f~a1+a2+a3+a4, cost=0.1, kmdat.train)

# linear SVR for g
svrg <- svr.linear(g~a1+a2+a3+a4, kmdat.train)
svrg.e1 <- svr.linear(g~a1+a2+a3+a4, eps=1, kmdat.train)
svrg.c01 <- svr.linear(g~a1+a2+a3+a4, cost=0.1, kmdat.train)

# linear SVR for the Boston Housing data
bh.svr <- svr.linear(medv~., bh.std.train)
bh.svr.e1 <- svr.linear(medv~., eps=1, bh.std.train)
bh.svr.c01 <- svr.linear(medv~., cost=0.1, bh.std.train)

# training set MSE
mse(predict(svrf$model, kmdat.train[,1:4]), kmdat.train$f)
mse(predict(svrf.e1$model, kmdat.train[,1:4]), kmdat.train$f)
mse(predict(svrf.c01$model, kmdat.train[,1:4]), kmdat.train$f)

mse(predict(svrg$model, kmdat.train[,1:4]), kmdat.train$g)
mse(predict(svrg.e1$model, kmdat.train[,1:4]), kmdat.train$g)
mse(predict(svrg.c01$model, kmdat.train[,1:4]), kmdat.train$g)

mse(predict(bh.svr$model, bh.std.train[,-13]), bh.std.train$medv)
mse(predict(bh.svr.e1$model, bh.std.train[,-13]), bh.std.train$medv)
mse(predict(bh.svr.c01$model, bh.std.train[,-13]), bh.std.train$medv)

# test set MSE
mse(predict(svrf$model, kmdat.test[,1:4]), kmdat.test$f)
mse(predict(svrf.e1$model, kmdat.test[,1:4]), kmdat.test$f)
mse(predict(svrf.c01$model, kmdat.test[,1:4]), kmdat.test$f)

mse(predict(svrg$model, kmdat.test[,1:4]), kmdat.test$g)
mse(predict(svrg.e1$model, kmdat.test[,1:4]), kmdat.test$g)
mse(predict(svrg.c01$model, kmdat.test[,1:4]), kmdat.test$g)

mse(predict(bh.svr$model, bh.std.test[,-13]), bh.std.test$medv)
mse(predict(bh.svr.e1$model, bh.std.test[,-13]), bh.std.test$medv)
mse(predict(bh.svr.c01$model, bh.std.test[,-13]), bh.std.test$medv)
##############
#16-3-3.R
##############
# eps=0.5, cost=1
svr.t.05.1 <- svr.linear(f~., kmdat.t, solver="ipop", eps=0.5, cost=1)
w.t.05.1 <- svr.t.05.1$model$w
l2norm(w.t.05.1[-length(w.t.05.1)])

# eps=0.5, cost=0.1
svr.t.05.01 <- svr.linear(f~., kmdat.t, solver="ipop", eps=0.5, cost=0.1)
w.t.05.01 <- svr.t.05.01$model$w
l2norm(w.t.05.01[-length(w.t.05.01)])

# eps=0.5, cost=0.01
svr.t.05.001 <- svr.linear(f~., kmdat.t, solver="ipop", eps=0.5, cost=0.01)
w.t.05.001 <- svr.t.05.001$model$w
l2norm(w.t.05.001[-length(w.t.05.001)])

# eps=1, cost=1
svr.t.1.1 <- svr.linear(f~., kmdat.t, solver="ipop", eps=1, cost=1)
w.t.1.1 <- svr.t.1.1$model$w
l2norm(w.t.1.1[-length(w.t.1.1)])

# eps=1, cost=0.1
svr.t.1.01 <- svr.linear(f~., kmdat.t, solver="ipop", eps=1, cost=0.1)
w.t.1.01 <- svr.t.1.01$model$w
l2norm(w.t.1.01[-length(w.t.1.01)])

# eps=1, cost=0.01
svr.t.1.001 <- svr.linear(f~., kmdat.t, solver="ipop", eps=1, cost=0.01)
w.t.1.001 <- svr.t.1.001$model$w
l2norm(w.t.1.001[-length(w.t.1.001)])

par(mfcol=c(3, 2))

plot.tube(w.t.05.1, kmdat.t, eps=0.5, main="eps=0.5, cost=1")
plot.tube(w.t.05.01, kmdat.t, eps=0.5, main="eps=0.5, cost=0.1")
plot.tube(w.t.05.001, kmdat.t, eps=0.5, main="eps=0.5, cost=0.01")

plot.tube(w.t.1.1, kmdat.t, eps=1, main="eps=1, cost=1")
plot.tube(w.t.1.01, kmdat.t, eps=1, main="eps=1, cost=0.1")
plot.tube(w.t.1.001, kmdat.t, eps=1, main="eps=1, cost=0.01")
##############
#16-4-1.R
##############
## data transformation that generates new attributes
## defined as the products of all original attribute pairs
trans.mult2 <- function(data)
{
  t(apply(data, 1, function(d) d %o% d))
}

# original dataset
kmdat.orig <- kmdat.train[1:10,1:4]
# dot product matrix for the original dataset
kmdat.dp <- as.matrix(kmdat.orig) %*% t(kmdat.orig)

# transformed dataset
kmdat.trans <- trans.mult2(kmdat.orig)
# dot product matrix for the transformed dataset
kmdat.dpt <- kmdat.trans %*% t(kmdat.trans)

# verify that the dot product matrix for the transformed dataset
# is the same as the squared original dot product matrix
max(abs((kmdat.dpt-kmdat.dp^2)))
##############
#16-5-1.R
##############
## can be called for both single attribute value vectors and for the whole dataset
kernel.linear <- function(av1, av2=av1) { as.matrix(av1)%*%t(av2) }

## can be called for both single attribute value vectors and for the whole dataset
kernel.polynomial <- function(av1, av2=av1, gamma=1, b=0, p=3)
{ (gamma*(as.matrix(av1)%*%t(av2))+b)^p }

## can be called for both single attribute value vectors and for the whole dataset
kernel.radial <- function(av1, av2=av1, gamma=1)
{
  exp(-gamma*outer(1:nrow(av1 <- as.matrix(av1)), 1:ncol(av2 <- t(av2)),
                   Vectorize(function(i, j) l2norm(av1[i,]-av2[,j])^2)))
}

## can be called for both single attribute value vectors and for the whole dataset
kernel.sigmoid <- function(av1, av2=av1, gamma=0.1, b=0)
{ tanh(gamma*(as.matrix(av1)%*%t(av2))+b) }

# kernel functions called for instance pairs
kernel.linear(kmdat.train[1,1:4], kmdat.train[2,1:4])
kernel.polynomial(kmdat.train[1,1:4], kmdat.train[2,1:4])
kernel.radial(kmdat.train[1,1:4], kmdat.train[2,1:4])
kernel.sigmoid(kmdat.train[1,1:4], kmdat.train[2,1:4])

# kernel functions called for the dataset (using the first 10 instances)
kernel.linear(kmdat.train[1:10,1:4])
kernel.polynomial(kmdat.train[1:10,1:4])
kernel.radial(kmdat.train[1:10,1:4])
kernel.sigmoid(kmdat.train[1:10,1:4])
##############
#16-6-1.R
##############
## predict using a kernel-based model
predict.kernel <- function(model, data)
{
  attributes <- x.vars(model$formula, data)
  aind <- names(data) %in% attributes
  amat <- as.matrix(data[,aind,drop=FALSE])
  kmat <- do.call(model$kernel, c(list(amat, model$mat), model$kernel.args))
  rowSums(cmm(kmat, model$coef))+model$intercept
}

# kernel models for producing plots
kmplot <- list(coef=c(rep(1, 50), rep(-2, 50)),
               mat=as.matrix(kmdat.plot[sample(nrow(kmdat.plot), 100),1:2]),
               intercept=1, formula=f~a1+a2)
kmplot.l <- `class<-`(c(kmplot, kernel=kernel.linear), "kernel")
kmplot.p <- `class<-`(c(kmplot, kernel=kernel.polynomial), "kernel")
kmplot.r <- `class<-`(c(kmplot, kernel=kernel.radial), "kernel")
kmplot.s <- `class<-`(c(kmplot, kernel=kernel.sigmoid), "kernel")

# generate predictions using different kernel functions
kmdat.plot$hl <- predict(kmplot.l, kmdat.plot)
kmdat.plot$hp <- predict(kmplot.p, kmdat.plot)
kmdat.plot$hr <- predict(kmplot.r, kmdat.plot)
kmdat.plot$hs <- predict(kmplot.s, kmdat.plot)

# plot prediction surfaces
wf.kl <- wireframe(hl~a1+a2, kmdat.plot, col="grey50", zoom=0.8)
wf.kp <- wireframe(hp~a1+a2, kmdat.plot, col="grey50", zoom=0.8)
wf.kr <- wireframe(hr~a1+a2, kmdat.plot, col="grey50", zoom=0.8)
wf.ks <- wireframe(hs~a1+a2, kmdat.plot, col="grey50", zoom=0.8)

print(wf.kl, split=c(1, 1, 2, 2), more=TRUE)
print(wf.kp, split=c(2, 1, 2, 2), more=TRUE)
print(wf.kr, split=c(1, 2, 2, 2), more=TRUE)
print(wf.ks, split=c(2, 2, 2, 2))
##############
#16-7-1.R
##############
## kernel-based soft-margin SVM parameter estimation using quadratic programming
## solvers: "solve.QP" or "ipop"
svm.kernel <- function(formula, data, kernel=kernel.linear, kernel.args=NULL,
                       cost=1, svthres=1e-3, solver="solve.QP")
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes
  
  cvec <- 2*as.num0(data[[class]])-1  # class vector using {-1, 1} labels
  ccmat <- outer(cvec, cvec)          # class-class product matrix
  amat <- as.matrix(data[,aind])      # attribute value matrix
  kmat <- do.call(kernel, c(list(amat), kernel.args)) # kernel matrix
  
  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(kmat*ccmat)$mat,
                 dvec=rep(1, nrow(data)),
                 Amat=matrix(c(cvec, diag(1, nrow(data)), diag(-1, nrow(data))),
                             nrow=nrow(data)),
                 bvec=c(0, rep(0, nrow(data)), rep(-cost, nrow(data))),
                 meq=1)
  else if (solver=="ipop")
    args <- list(c=rep(-1, nrow(data)),
                 H=kmat*ccmat,
                 A=cvec,
                 b=0,
                 l=rep(0, nrow(data)),
                 u=rep(cost, nrow(data)),
                 r=0)
  else
    stop("Unknown solver: ", solver)
  
  qp <- do.call(solver, args)
  alpha <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  sv <- which(alpha>svthres)
  model <- list(coef=cvec[sv]*alpha[sv], mat=amat[sv,,drop=FALSE],
                kernel=kernel, kernel.args=kernel.args, formula=formula)
  i <- which.min(abs(alpha-cost/2))
  'class<-'(c(model, intercept=cvec[i]-
                unname(predict.kernel(c(model, intercept=0),
                                      data[i,aind,drop=FALSE]))),
            "svm.kernel")
}

## kernel-based SVM prediction
predict.svm.kernel <- function(model, data)
{
  ustep(predict.kernel(model, data))
}

# kernel-based SVM for the artificial data
svm.kl <- svm.kernel(c~a1+a2+a3+a4, kmdat.train)
svm.kp <- svm.kernel(c~a1+a2+a3+a4, kmdat.train,
                     kernel=kernel.polynomial, kernel.args=list(p=2, b=1))
svm.kr <- svm.kernel(c~a1+a2+a3+a4, kmdat.train,
                     kernel=kernel.radial, kernel.args=list(gamma=0.5))
svm.ks <- svm.kernel(c~a1+a2+a3+a4, kmdat.train,
                     kernel=kernel.sigmoid, kernel.args=list(gamma=0.04,b=-0.8))

# kernel-based SVM for the Pima Indians Diabetes
pid.svm.kl <- svm.kernel(diabetes~., pid.std.train)
pid.svm.kr <- svm.kernel(diabetes~., pid.std.train,
                         kernel=kernel.radial, kernel.args=list(gamma=0.1))

# training set misclassification error
err(predict(svm.kl, kmdat.train), kmdat.train$c)
err(predict(svm.kp, kmdat.train), kmdat.train$c)
err(predict(svm.kr, kmdat.train), kmdat.train$c)
err(predict(svm.ks, kmdat.train), kmdat.train$c)

err(factor(predict(pid.svm.kl, pid.std.train[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.std.train$diabetes)
err(factor(predict(pid.svm.kr, pid.std.train[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.std.train$diabetes)

# test set misclassification error
err(predict(svm.kl, kmdat.test), kmdat.test$c)
err(predict(svm.kp, kmdat.test), kmdat.test$c)
err(predict(svm.kr, kmdat.test), kmdat.test$c)
err(predict(svm.ks, kmdat.test), kmdat.test$c)

err(factor(predict(pid.svm.kl, pid.std.test[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.test$diabetes)
err(factor(predict(pid.svm.kr, pid.std.test[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.test$diabetes)
##############
#16-7-2.R
##############
## kernel-based SVR parameter estimation using quadratic programming
## solvers: "solve.QP" or "ipop"
svr.kernel <- function(formula, data, eps=0.01,
                       kernel=kernel.linear, kernel.args=NULL,
                       cost=1, svthres=1e-3, solver="solve.QP")
{
  f <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes
  
  fvec <- data[[f]]                   # target function vector
  amat <- as.matrix(data[,aind])      # attribute value matrix
  kmat <- do.call(kernel, c(list(amat), kernel.args)) # kernel matrix
  
  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(rbind(cbind(kmat, -kmat), cbind(-kmat, kmat)))$mat,
                 dvec=c(fvec-eps, -fvec-eps),
                 Amat=matrix(c(rep(1, nrow(data)), rep(-1, nrow(data)),
                               diag(1, 2*nrow(data)), diag(-1, 2*nrow(data))),
                             nrow=2*nrow(data)),
                 bvec=c(0, rep(0, 2*nrow(data)), rep(-cost, 2*nrow(data))),
                 meq=1)
  else if (solver=="ipop")
    args <- list(c=c(-fvec+eps, fvec+eps),
                 H=rbind(cbind(kmat, -kmat), cbind(-kmat, kmat)),
                 A=c(rep(1, nrow(data)), rep(-1, nrow(data))),
                 b=0,
                 l=rep(0, 2*nrow(data)),
                 u=rep(cost, 2*nrow(data)),
                 r=0)
  else
    stop("Unknown solver: ", solver)
  
  qp <- do.call(solver, args)
  alpha <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  beta <- alpha[1:nrow(data)]-alpha[(nrow(data)+1):(2*nrow(data))]
  sv <- which(abs(beta)>svthres)
  model <- list(coef=beta[sv], mat=amat[sv,,drop=FALSE],
                kernel=kernel, kernel.args=kernel.args, formula=formula)
  i <- which.min(abs(beta-cost/2))
  `class<-`(c(model,
              intercept=fvec[i]-unname(predict.kernel(c(model, intercept=0),
                                                      data[i,aind,drop=FALSE]))-
                sign(beta[i])*eps),
            "svr.kernel")
}

## kernel-based SVR prediction
predict.svr.kernel <- predict.kernel

# kernel-based SVR for f
svrf.kl <- svr.kernel(f~a1+a2+a3+a4, kmdat.train)
svrf.kp <- svr.kernel(f~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.polynomial, kernel.args=list(p=2, b=1))
svrf.kr <- svr.kernel(f~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.radial, kernel.args=list(gamma=0.02))
svrf.ks <- svr.kernel(f~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.sigmoid, kernel.args=list(gamma=0.2, b=0))

# kernel-based SVR for g
svrg.kl <- svr.kernel(g~a1+a2+a3+a4, kmdat.train)
svrg.kp <- svr.kernel(g~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.polynomial, kernel.args=list(p=2, b=1))
svrg.kr <- svr.kernel(g~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.radial, kernel.args=list(gamma=0.1))
svrg.ks <- svr.kernel(g~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.sigmoid, kernel.args=list(gamma=0.02, b=-1))

# kernel-based SVR for the Boston Housing data
bh.svr.kl <- svr.kernel(medv~., bh.std.train)
bh.svr.kr <- svr.kernel(medv~., bh.std.train,
                        kernel=kernel.radial, kernel.args=list(gamma=0.1))

# training set MSE
mse(predict(svrf.kl, kmdat.train), kmdat.train$f)
mse(predict(svrf.kp, kmdat.train), kmdat.train$f)
mse(predict(svrf.kr, kmdat.train), kmdat.train$f)
mse(predict(svrf.ks, kmdat.train), kmdat.train$f)

mse(predict(svrg.kl, kmdat.train), kmdat.train$g)
mse(predict(svrg.kp, kmdat.train), kmdat.train$g)
mse(predict(svrg.kr, kmdat.train), kmdat.train$g)
mse(predict(svrg.ks, kmdat.train), kmdat.train$g)

mse(predict(bh.svr.kl, bh.std.train[,-13]), bh.std.train$medv)
mse(predict(bh.svr.kr, bh.std.train[,-13]), bh.std.train$medv)

# test set MSE
mse(predict(svrf.kl, kmdat.test), kmdat.test$f)
mse(predict(svrf.kp, kmdat.test), kmdat.test$f)
mse(predict(svrf.kr, kmdat.test), kmdat.test$f)
mse(predict(svrf.ks, kmdat.test), kmdat.test$f)

mse(predict(svrg.kl, kmdat.test), kmdat.test$g)
mse(predict(svrg.kp, kmdat.test), kmdat.test$g)
mse(predict(svrg.kr, kmdat.test), kmdat.test$g)
mse(predict(svrg.ks, kmdat.test), kmdat.test$g)

mse(predict(bh.svr.kl, bh.std.test[,-13]), bh.std.test$medv)
mse(predict(bh.svr.kr, bh.std.test[,-13]), bh.std.test$medv)