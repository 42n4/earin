##############
#7-1-1.R
##############
library(dmr.util)
library(rpart)

data(Soybean, package="mlbench")

set.seed(12)
rs <- runif(nrow(Soybean))
s.train <- Soybean[rs>=0.33,]
s.test <- Soybean[rs<0.33,]

s.tree <- rpart(Class~., s.train)
##############
#7-2-1.R
##############
err <- function(pred.y, true.y) { mean(pred.y!=true.y) }

err(predict(s.tree, s.test, type="c"), s.test$Class)
##############
#7-2-2.R
##############
werr <- function(pred.y, true.y, w=rep(1, length(true.y)))
{ weighted.mean(pred.y!=true.y, w) }

# double weight for the least frequent class
s.w2test <- ifelse(s.test$Class=="herbicide-injury", 2, 1)
werr(predict(s.tree, s.test, type="c"), s.test$Class, s.w2test)

# random per-class weights 1..5
s.wctest <- round(runif(nlevels(s.test$Class), min=1, max=5))
s.w3test <- s.wctest[s.test$Class]
werr(predict(s.tree, s.test, type="c"), s.test$Class, s.w3test)
##############
#7-2-3.R
##############
mean.cost <- function(pred.y, true.y, rho) { mean(diag(rho[pred.y,true.y])) }

# uniform cost matrix
s.r1test <- matrix(1, nrow=nlevels(s.test$Class), ncol=nlevels(s.test$Class))
diag(s.r1test) <- 0
mean.cost(predict(s.tree, s.test, type="c"), s.test$Class, s.r1test)

# double cost for misclassifying the least frequent class
s.r2test <- matrix(1, nrow=nlevels(s.test$Class), ncol=nlevels(s.test$Class))
s.r2test[,levels(s.test$Class)=="herbicide-injury"] <- 2
diag(s.r2test) <- 0
mean.cost(predict(s.tree, s.test, type="c"), s.test$Class, s.r2test)
# this should give the same result
sum(s.w2test)/nrow(s.test)*werr(predict(s.tree, s.test, type="c"),
                                s.test$Class, s.w2test)

# random per-class costs 1..5
s.r3test <- matrix(s.wctest, nrow=nlevels(s.test$Class), ncol=nlevels(s.test$Class),
                   byrow=TRUE)
diag(s.r3test) <- 0
mean.cost(predict(s.tree, s.test, type="c"), s.test$Class, s.r3test)
# this should give the same result
sum(s.w3test)/nrow(s.test)*werr(predict(s.tree, s.test, type="c"),
                                s.test$Class, s.w3test)

# random costs 1..5
s.r4test <- matrix(round(runif(nlevels(s.test$Class)*nlevels(s.test$Class),
                               min=1, max=5)),
                   nrow=nlevels(s.test$Class), ncol=nlevels(s.test$Class))
diag(s.r4test) <- 0
mean.cost(predict(s.tree, s.test, type="c"), s.test$Class, s.r4test)
##############
#7-2-4.R
##############
confmat <- function(pred.y, true.y)
{ table(pred.y, true.y, dnn=c("predicted", "true")) }

s.cm <- confmat(predict(s.tree, s.test, type="c"), s.test$Class)
# error
(sum(s.cm)-sum(diag(s.cm)))/(sum(s.cm))
# mean misclassification cost
sum(s.cm*s.r4test)/sum(s.cm)
##############
#7-2-5.R
##############
s01.labels <- c("other", "brown-spot")
Soybean01 <- Soybean
Soybean01$Class <- factor(ifelse(Soybean$Class=="brown-spot", "brown-spot", "other"),
                          levels=s01.labels)

s01.train <- Soybean01[rs>=0.33,]
s01.test <- Soybean01[rs<0.33,]

s01.tree <- rpart(Class~., s01.train)
##############
#7-2-6.R
##############
tpr <- function(cm) { if (is.nan(p <- cm[2,2]/(cm[2,2]+cm[1,2]))) 1 else p }
fpr <- function(cm) { if (is.nan(p <- cm[2,1]/(cm[2,1]+cm[1,1]))) 1 else p }
precision <- function(cm) { if (is.nan(p <- cm[2,2]/(cm[2,2]+cm[2,1]))) 1 else p }
recall <- tpr
sensitivity <- tpr
specificity <- function(cm) { if (is.nan(p <- cm[1,1]/(cm[2,1]+cm[1,1]))) 1 else p }

s01.cm <- confmat(predict(s01.tree, s01.test, type="c"), s01.test$Class)

list(tpr=tpr(s01.cm),
     fpr=fpr(s01.cm),
     precision=precision(s01.cm),
     recall=recall(s01.cm),
     sensitivity=sensitivity(s01.cm),
     specificity=specificity(s01.cm))
##############
#7-2-7.R
##############
f.measure <- function(cm) { 1/mean(c(1/precision(cm), 1/recall(cm))) }

f.measure(s01.cm)
##############
#7-2-8.R
##############
## per-class 1 vs. rest confusion matrices
confmat01 <- function(pred.y, true.y)
{
  `names<-`(lapply(levels(true.y),
                   function(d)
                   {
                     cm <- confmat(factor(as.integer(pred.y==d), levels=0:1),
                                   factor(as.integer(true.y==d), levels=0:1))
                   }), levels(true.y))
}

s.cm01 <- confmat01(predict(s.tree, s.test, type="c"), s.test$Class)
# average TP rate, FP rate, and f-measure
rowMeans(sapply(s.cm01, function(cm) c(tpr=tpr(cm), fpr=fpr(cm), fm=f.measure(cm))))
##############
#7-2-9.R
##############
wconfmat <- function(pred.y, true.y, w=rep(1, length(true.y)))
{ weighted.table(pred.y, true.y, w=w, dnn=c("predicted", "true")) }

# double weight for the brown-spot class
s01.w1test <- ifelse(s01.test$Class=="brown-spot", 2, 1)
s01.w1cm <- wconfmat(predict(s01.tree, s01.test, type="c"),
                     s01.test$Class, s01.w1test)
tpr(s01.w1cm)
fpr(s01.w1cm)

# 10 times less weight for instances with plant.stand=1
s01.w2test <- ifelse(!is.na(s01.test$plant.stand) & s01.test$plant.stand=="1",
                     0.1, 1)
s01.w2cm <- wconfmat(predict(s01.tree, s01.test, type="c"),
                     s01.test$Class, s01.w2test)
tpr(s01.w2cm)
fpr(s01.w2cm)

##############
#7-2-10.R
##############
# predicted scores and true class labels
sctab <- data.frame(score=c(1, 1, 3, 4, 5, 5, 6, 7, 9, 9),
                    class=factor(c(0, 0, 1, 0, 0, 1, 1 ,0, 1, 1)))

# operating point identification
scroc <- t(sapply(c(sort(unique(sctab$score)), Inf),
                  function(sc)
                  {
                    pred <- factor(as.numeric(sctab$score<sc),
                                   levels=levels(sctab$class))
                    list(tpr=tpr(cm <- confmat(pred, sctab$class)),
                         fpr=fpr(cm))
                  }))

# ROC curve
plot(scroc, type="l", xlab="FP rate", ylab="TP rate")
##############
#7-2-11.R
##############
pred.s<-as.vector(predict(s01.tree, s01.test)[,2])
true.y<-s01.test$Class
roc <- function(pred.s, true.y)
{
  cutoff <- Inf  # start with all instances classified as negative
  tp <- fp <- 0
  tn <- sum(2-as.integer(true.y))  # all negative instances
  fn <- sum(as.integer(true.y)-1)  # all positive instances
  rt <- data.frame()
  
  sord <- order(pred.s, decreasing=TRUE)  # score ordering
  #pred.s[sord]
  for (i in 1:length(sord))
  {
    if (pred.s[sord[i]] < cutoff)
    {
      rt <- rbind(rt, data.frame(tpr=tp/(tp+fn), fpr=fp/(fp+tn), cutoff=cutoff))
      cutoff <- pred.s[sord[i]]
    }
    
    p <- as.integer(true.y[sord[i]])-1  # next positive classified as positive
    n <- 2-as.integer(true.y[sord[i]])  # next negative classified as positive
    tp <- tp+p
    fp <- fp+n
    tn <- tn-n
    fn <- fn-p
  }
  rt <- rbind(rt, data.frame(tpr=tp/(tp+fn), fpr=fp/(fp+tn), cutoff=cutoff))
}

# ROC curve for the decision tree model
s01.roc <- roc(predict(s01.tree, s01.test)[,2], s01.test$Class)
plot(s01.roc$fpr, s01.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")

# ROC curve for a random model
s01rand <- runif(nrow(s01.test))
s01rand.roc <- roc(s01rand, s01.test$Class)
lines(s01rand.roc$fpr, s01rand.roc$tpr, lty=2)
##############
#7-2-12.R
##############
## assign class labels according to the given cutoff value
cutclass <- function(s, cutoff, labels) { factor(ustep(s, cutoff), labels=labels) }

## identify the best cutoff value
## satisfying the minimum tpr or maximum fpr constraint
roc.shift <- function(r, min.tpr=NULL, max.fpr=NULL)
{
  if (!is.null(min.tpr))
    max(r$cutoff[r$tpr>=min.tpr])
  else if (!is.null(max.fpr))
    min(r$cutoff[r$fpr<=max.fpr])
  else
    0.5
}

# shift to achieve tpr>0.85 at minimum fpr
s01.t085 <- roc.shift(s01.roc, min.tpr=0.85)
s01.cm.t085 <- confmat(cutclass(predict(s01.tree, s01.test)[,2],
                                s01.t085, s01.labels),
                       s01.test$Class)
# shift to achieve maximum tpr at fpr<0.5
s01.f05 <- roc.shift(s01.roc, max.fpr=0.5)
s01.cm.f05 <- confmat(cutclass(predict(s01.tree, s01.test)[,2], s01.f05, s01.labels),
                      s01.test$Class)
# the ROC curve
plot(s01.roc$fpr, s01.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
# the default operating point
points(fpr(s01.cm), tpr(s01.cm), pch=8)
# the shifted operating points
points(fpr(s01.cm.t085), tpr(s01.cm.t085), pch=1)
points(fpr(s01.cm.f05), tpr(s01.cm.f05), pch=2)
##############
#7-2-13.R
##############
mixclass <- function(c1, c2, p)
{ factor(ifelse(p<runif(length(c1)), c1, c2), labels=levels(c1)) }

# interpolate between the two shifted operating points
s01.mix <- mixclass(cutclass(predict(s01.tree, s01.test)[,2], s01.t085, s01.labels),
                    cutclass(predict(s01.tree, s01.test)[,2], s01.f05, s01.labels),
                    0.75)
s01.cmi <- confmat(s01.mix, s01.test$Class)

# the ROC curve
plot(s01.roc$fpr, s01.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
# the default operating point
points(fpr(s01.cm), tpr(s01.cm), pch=8)
# the 1st shifted operating point
points(fpr(s01.cm.t085), tpr(s01.cm.t085), pch=1)
# the 2nd shifted operating point
points(fpr(s01.cm.f05), tpr(s01.cm.f05), pch=2)
# the interpolated operating point
points(fpr(s01.cmi), tpr(s01.cmi), pch=5)
##############
#7-2-14.R
##############
auc <- function(roc)
{ n <- nrow(roc); sum((roc$tpr[1:n-1]+roc$tpr[2:n])*diff(roc$fpr)/2) }

# area under the ROC curve for the decision tree model
auc(s01.roc)
# area under the ROC curve for a random model
auc(s01rand.roc)
##############
#7-2-15.R
##############
wroc <- function(pred.s, true.y, w=rep(1, length(true.y)))
{
  cutoff <- Inf  # start with all instances classified as negative
  tp <- fp <- 0
  tn <- sum((2-as.integer(true.y))*w)  # all negative instances
  fn <- sum((as.integer(true.y)-1)*w)  # all positive instances
  rt <- data.frame()
  
  sord <- order(pred.s, decreasing=TRUE)  # score ordering
  for (i in 1:length(sord))
  {
    if (pred.s[sord[i]] < cutoff)
    {
      rt <- rbind(rt, data.frame(tpr=tp/(tp+fn), fpr=fp/(fp+tn), cutoff=cutoff))
      cutoff <- pred.s[sord[i]]
    }
    
    p <- (as.integer(true.y[sord[i]])-1)*w[sord[i]]  # next positive
    n <- (2-as.integer(true.y[sord[i]]))*w[sord[i]]  # next negative
    tp <- tp+p
    fp <- fp+n
    tn <- tn-n
    fn <- fn-p
  }
  rt <- rbind(rt, data.frame(tpr=tp/(tp+fn), fpr=fp/(fp+tn), cutoff=cutoff))
}

# ROC curve with double weight for the brown-spot class
s01.w1roc <- wroc(predict(s01.tree, s01.test)[,2], s01.test$Class, s01.w1test)
plot(s01.w1roc$fpr, s01.w1roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
auc(s01.w1roc)

# ROC curve with 10 times less weight for instances with plant.stand=1
s01.w2roc <- wroc(predict(s01.tree, s01.test)[,2], s01.test$Class, s01.w2test)
lines(s01.w2roc$fpr, s01.w2roc$tpr, lty=2)
legend("bottomright", c("brown-spot x2", "plant.stand=1 x10"), lty=1:2)
auc(s01.w2roc)
##############
#7-2-16.R
##############
## likelihood for probabilistic classifier evaluation
## assuming eps for 0 probabilities
lik <- function(prob.y, true.y, eps=.Machine$double.eps)
{
  prod(pmax(sapply(1:length(tn <- as.numeric(true.y)),
                   function(i) prob.y[i,tn[i]]), eps))
}

## likelihood for probabilistic binary classifier evaluation
## assuming eps for 0 probabilities
lik01 <- function(prob.y, true.y, eps=.Machine$double.eps)
{
  prod((py <- pmin(pmax(prob.y, eps), 1-eps))^(t01 <- as.num0(true.y))*(1-py)^(1-t01))
}

# likelihood for the Soybean data
lik(predict(s.tree, s.test), s.test$Class)
lik(predict(s01.tree, s01.test), s01.test$Class)
lik01(predict(s01.tree, s01.test)[,2], s01.test$Class)
##############
#7-2-17.R
##############
## loglikelihood for probabilistic classifier evaluation
## assuming eps for 0 probabilities
loglik <- function(prob.y, true.y, eps=.Machine$double.eps)
{
  sum(log(pmax(sapply(1:length(tn <- as.numeric(true.y)),
                      function(i) prob.y[i,tn[i]]), eps)))
}

## loglikelihood for probabilistic binary classifier evaluation
## assuming eps for 0 probabilities
loglik01 <- function(prob.y, true.y, eps=.Machine$double.eps)
{
  sum((t01 <- as.num0(true.y))*log(py <- pmin(pmax(prob.y, eps), 1-eps))+
        (1-t01)*log(1-py))
}

# loglikelihood for the Soybean data
loglik(predict(s.tree, s.test), s.test$Class)
loglik(predict(s01.tree, s01.test), s01.test$Class)
loglik01(predict(s01.tree, s01.test)[,2], s01.test$Class)
##############
#7-3-1.R
##############
holdout <- function(alg, formula, data, args=NULL, predf=predict, prob=FALSE,
                    p=0.33, n=1)
{
  yn <- as.character(formula)[2]  # class column name
  ylabs <- levels(data[[yn]])     # class labels
  pred.y <- NULL  # predictions
  true.y <- NULL  # true class labels
  
  for (t in 1:n)
  {
    r <- runif(nrow(data))
    train <- data[r>=p,]
    test <- data[r<p,]
    model <- do.call(alg, c(list(formula, train), args))
    pred.y <- c(pred.y, predf(model, test))
    true.y <- c(true.y, test[[yn]])
  }
  
  if (!is.null(ylabs))
  {
    if (!prob)
      pred.y <- factor(pred.y, levels=1:length(ylabs), labels=ylabs)
    true.y <- factor(true.y, levels=1:length(ylabs), labels=ylabs)
  }
  
  return(data.frame(pred=pred.y, true=true.y))
}

# hold-out evaluation of discrete predictions
s01ho <- holdout(rpart, Class~., Soybean01,
                 predf=function(...) predict(..., type="c"), n=10)
err(s01ho$pred, s01ho$true)
confmat(s01ho$pred, s01ho$true)

# hold-out evaluation of probabilistic predictions
s01hop <- holdout(rpart, Class~., Soybean01,
                  predf=function(...) predict(..., type="p")[,2], prob=TRUE, n=10)
s01hop.roc <- roc(s01hop$pred, s01hop$true)
plot(s01hop.roc$fpr, s01hop.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
auc(s01hop.roc)
##############
#7-3-2.R
##############
crossval <- function(alg, formula, data, args=NULL, predf=predict, prob=FALSE,
                     k=10, n=1)
{
  yn <- as.character(formula)[2]  # class column name
  ylabs <- levels(data[[yn]])     # class labels
  pred.y <- NULL  # predictions
  true.y <- NULL  # true class labels
  
  for (t in 1:n)
  {
    ind <- sample(k, size=nrow(data), replace=TRUE)  # index of k random subsets
    for (i in 1:k)
    {
      train <- data[ind!=i,]
      test <- data[ind==i,]
      model <- do.call(alg, c(list(formula, train), args))
      pred.y <- c(pred.y, predf(model, test))
      true.y <- c(true.y, test[[yn]])
    }
  }
  
  if (!is.null(ylabs))
  {
    if (!prob)
      pred.y <- factor(pred.y, levels=1:length(ylabs), labels=ylabs)
    true.y <- factor(true.y, levels=1:length(ylabs), labels=ylabs)
  }
  
  return(data.frame(pred=pred.y, true=true.y))
}

# 3-fold cross-validation for discrete predictions
s01cv3 <- crossval(rpart, Class~., Soybean01,
                   predf=function(...) predict(..., type="c"), k=3)
err(s01cv3$pred, s01cv3$true)
confmat(s01cv3$pred, s01cv3$true)

# 10-fold cross-validation for discrete predictions
s01cv10 <- crossval(rpart, Class~., Soybean01,
                    predf=function(...) predict(..., type="c"), k=10)
err(s01cv10$pred, s01cv10$true)
confmat(s01cv10$pred, s01cv10$true)

# 20-fold cross-validation for discrete predictions
s01cv20 <- crossval(rpart, Class~., Soybean01,
                    predf=function(...) predict(..., type="c"), k=20)
err(s01cv20$pred, s01cv20$true)
confmat(s01cv20$pred, s01cv20$true)

# 4x5-fold cross-validation for discrete predictions
s01cv4x5 <- crossval(rpart, Class~., Soybean01,
                     predf=function(...) predict(..., type="c"), k=5, n=4)
err(s01cv20$pred, s01cv4x5$true)
confmat(s01cv4x5$pred, s01cv4x5$true)

# 10-fold cross-validation for probabilistic predictions
s01cv10p <- crossval(rpart, Class~., Soybean01,
                     predf=function(...) predict(..., type="p")[,2], prob=TRUE, k=10)
s01cv10p.roc <- roc(s01cv10p$pred, s01cv10p$true)
plot(s01cv10p.roc$fpr, s01cv10p.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
auc(s01cv10p.roc)
##############
#7-3-3.R
##############
leave1out <- function(alg, formula, data, args=NULL, predf=predict, prob=FALSE)
{
  yn <- as.character(formula)[2]  # class column name
  ylabs <- levels(data[[yn]])     # class labels
  pred.y <- NULL  # predictions
  true.y <- NULL  # true class labels
  
  for (i in 1:nrow(data))
  {
    train <- data[-i,]
    test <- data[i,]
    model <- do.call(alg, c(list(formula, train), args))
    pred.y <- c(pred.y, predf(model, test))
    true.y <- c(true.y, test[[yn]])
  }
  
  if (!is.null(ylabs))
  {
    if (!prob)
      pred.y <- factor(pred.y, levels=1:length(ylabs), labels=ylabs)
    true.y <- factor(true.y, levels=1:length(ylabs), labels=ylabs)
  }
  
  return(data.frame(pred=pred.y, true=true.y))
}

# leave-one-out for discrete predictions
s01l1o <- leave1out(rpart, Class~., Soybean01,
                    predf=function(...) predict(..., type="c"))
err(s01l1o$pred, s01l1o$true)
confmat(s01l1o$pred, s01l1o$true)

# leave-one-out for probabilistic predictions
s01l1op <- leave1out(rpart, Class~., Soybean01,
                     predf=function(...) predict(..., type="p")[,2], prob=TRUE)
s01l1op.roc <- roc(s01l1op$pred, s01l1op$true)
plot(s01l1op.roc$fpr, s01l1op.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
auc(s01l1op.roc)
##############
#7-3-4.R
##############
bootstrap <- function(alg, formula, data, args=NULL, predf=predict, prob=FALSE,
                      w=0.632, m=100)
{
  yn <- as.character(formula)[2]  # class column name
  ylabs <- levels(data[[yn]])     # class labels
  pred.y.w <- NULL  # predictions
  true.y.w <- NULL  # true class labels
  
  for (t in 1:m)
  {
    bag <- sample(nrow(data), size=nrow(data), replace=TRUE)
    train <- data[bag,]
    test <- data[-bag,]
    model <- do.call(alg, c(list(formula, train), args))
    pred.y.w <- c(pred.y.w, predf(model, test))
    true.y.w <- c(true.y.w, test[[yn]])
  }
  
  if (w<1)
  {
    model <- do.call(alg, c(list(formula, data), args))
    pred.y.1w <- predf(model, data)
    true.y.1w <- data[[yn]]
    w <- c(rep(w/m, length(pred.y.w)), rep(1-w, nrow(data)))
  }
  else
  {
    pred.y.1w <- true.y.1w <- NULL
    w <- rep(w/m, length(pred.y.w))
  }
  
  pred.y <- c(pred.y.w, pred.y.1w)
  true.y <- c(true.y.w, true.y.1w)
  
  if (!is.null(ylabs))
  {
    if (!prob)
      pred.y <- factor(pred.y, levels=1:length(ylabs), labels=ylabs)
    true.y <- factor(true.y, levels=1:length(ylabs), labels=ylabs)
  }
  
  return(data.frame(pred=pred.y, true=true.y, w=w))
}

# 20x bootstrap for discrete predictions
s01bs20 <- bootstrap(rpart, Class~., Soybean01,
                     predf=function(...) predict(..., type="c"), w=1, m=20)
err(s01bs20$pred, s01bs20$true)
confmat(s01bs20$pred, s01bs20$true)

# 20x .632 bootstrap for discrete predictions
s01.632bs20 <- bootstrap(rpart, Class~., Soybean01,
                         predf=function(...) predict(..., type="c"), m=20)
# weighted error
werr(s01.632bs20$pred, s01.632bs20$true, s01.632bs20$w)
# weighted confusion matrix
wconfmat(s01.632bs20$pred, s01.632bs20$true, s01.632bs20$w)

# 20x bootstrap for probabilistic predictions
s01bs20p <- bootstrap(rpart, Class~., Soybean01,
                      predf=function(...) predict(..., type="p")[,2], prob=TRUE,
                      w=1, m=20)
s01bs20p.roc <- roc(s01bs20p$pred, s01bs20p$true)
plot(s01bs20p.roc$fpr, s01bs20p.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
auc(s01bs20p.roc)

# 20x .632 bootstrap for probabilistic predictions
s01.632bs20p <- bootstrap(rpart, Class~., Soybean01,
                          predf=function(...) predict(..., type="p")[,2], prob=TRUE,
                          m=20)
# weighted ROC
s01.632bs20p.roc <- wroc(s01.632bs20p$pred, s01.632bs20p$true, s01.632bs20p$w)
lines(s01.632bs20p.roc$fpr, s01.632bs20p.roc$tpr, lty=2)
legend("bottomright", c("plain", ".632"), lty=1:2)
auc(s01.632bs20p.roc)
##############
#7-3-5.R
##############
eval.bias.var <- function(alg, formula, data, args=NULL, predf=predict,
                          perf=err, wperf=werr, p=0.66, n=100)
{
  yn <- as.character(formula)[2]  # class column name
  performance <- data.frame()
  for (i in 1:n)
  {
    r <- runif(nrow(data))
    data.avail <- data[r<p,]   # pretend this is the available dataset
    data.new <- data[r>=0.7,]  # pretend this a new dataset
    model <- do.call(alg, c(list(formula, data.avail), args))
    
    cv3 <- crossval(alg, formula, data.avail, args=args, predf=predf, k=3)
    cv5 <- crossval(alg, formula, data.avail, args=args, predf=predf, k=5)
    cv10 <- crossval(alg, formula, data.avail, args=args, predf=predf, k=10)
    cv20 <- crossval(alg, formula, data.avail, args=args, predf=predf, k=20)
    cv5x4 <- crossval(alg, formula, data.avail, args=args, predf=predf, k=5, n=4)
    ho <- holdout(alg, formula, data.avail, args=args, predf=predf)
    hox10 <- holdout(alg, formula, data.avail, args=args, predf=predf, n=10)
    l1o <- leave1out(alg, formula, data.avail, args=args, predf=predf)
    bs10 <- bootstrap(alg, formula, data.avail, args=args, predf=predf, w=1, m=10)
    bs50 <- bootstrap(alg, formula, data.avail, args=args, predf=predf, w=1, m=50)
    bs10.632 <- bootstrap(alg, formula, data.avail, args=args, predf=predf, m=10)
    bs50.632 <- bootstrap(alg, formula, data.avail, args=args, predf=predf, m=50)
    
    performance <- rbind(performance,
                         data.frame(perf(predf(model, data.new), data.new[[yn]]),
                                    perf(cv3$pred, cv3$true),
                                    perf(cv5$pred, cv5$true),
                                    perf(cv10$pred, cv10$true),
                                    perf(cv20$pred, cv20$true),
                                    perf(cv5x4$pred, cv5x4$true),
                                    perf(ho$pred, ho$true),
                                    perf(hox10$pred, hox10$true),
                                    perf(l1o$pred, l1o$true),
                                    perf(bs10$pred, bs10$true),
                                    perf(bs50$pred, bs50$true),
                                    wperf(bs10.632$pred, bs10.632$true, bs10.632$w),
                                    wperf(bs50.632$pred, bs50.632$true, bs50.632$w)))
  }
  
  names(performance) <- c("true", "3-CV", "5-CV", "10-CV", "20-CV", "4x5-CV",
                          "HO", "10xHO", "L1O", "10-BS", "50-BS",
                          "10-BS.632", "50-BS.632")
  bias <- apply(performance[,-1]-performance[,1], 2, mean)
  variance <- apply(performance[,-1], 2, var)
  
  list(performance=performance, bias=bias, variance=variance)
}

# the commented lines run a 200-repetition experiment, which takes a long time
#s01.ebv <- eval.bias.var(rpart, Class~., Soybean01,
#                         predf=function(...) predict(..., type="c"), n=200)
# this can be used for a quick illustration
s01.ebv <- eval.bias.var(rpart, Class~., Soybean01,
                         predf=function(...) predict(..., type="c"), n=10)

boxplot(s01.ebv$performance[,-1], main="Error", las=2)
lines(c(0, 13), rep(mean(s01.ebv$performance[,1]), 2))
barplot(s01.ebv$bias, main="Bias", las=2)
barplot(s01.ebv$variance, main="Variance", las=2)
