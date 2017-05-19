library(dmr.claseval)
library(dmr.dectree)
library(dmr.regeval)
library(dmr.regtree)
library(dmr.stats)
library(dmr.util)

library(rpart)
library(e1071)

data(HouseVotes84, package="mlbench")
data(BostonHousing, package="mlbench")

set.seed(12)

rhv <- runif(nrow(HouseVotes84))
hv.train <- HouseVotes84[rhv>=0.33,]
hv.test <- HouseVotes84[rhv<0.33,]

rbh <- runif(nrow(BostonHousing))
bh.train <- BostonHousing[rbh>=0.33,]
bh.test <- BostonHousing[rbh<0.33,]

hv.tree <- rpart(Class~., hv.train)
hv.nb <- naiveBayes(Class~., hv.train)

hv.err.tree <- err(predict(hv.tree, hv.test, type="c"), hv.test$Class)
hv.err.nb <- err(predict(hv.nb, hv.test), hv.test$Class)

bh.tree <- rpart(medv~., bh.train)
bh.lm <- lm(medv~., bh.train)

bh.mse.tree <- mse(predict(bh.tree, bh.test), bh.test$medv)
bh.mse.lm <- mse(predict(bh.lm, bh.test), bh.test$medv)
##############
#15-3-1.R
##############
## generate base models by instance sampling
base.ensemble.sample.x <- function(formula, data, m, alg, args=NULL,
                                   size=nrow(data), replace=TRUE)
{
  lapply(1:m, function(i)
  {
    bag <- sample(nrow(data), size=nrow(data), replace=replace)
    do.call(alg, c(list(formula, data[bag,]), args))
  })
}

# base models for the HouseVotes84 data
hv.bm.tree.sx <- base.ensemble.sample.x(Class~., hv.train, 50, rpart)
hv.bm.nb.sx <- base.ensemble.sample.x(Class~., hv.train, 50, naiveBayes)

# base models for the BostonHousing data
bh.bm.tree.sx <- base.ensemble.sample.x(medv~., bh.train, 50, rpart)
bh.bm.lm.sx <- base.ensemble.sample.x(medv~., bh.train, 50, lm)

# base model training set errors for the HouseVotes84 data
hv.train.err.tree.sx <- sapply(hv.bm.tree.sx,
                               function(h) err(predict(h, hv.train, type="c"),
                                               hv.train$Class))
hv.train.err.nb.sx <- sapply(hv.bm.nb.sx,
                             function(h) err(predict(h, hv.train), hv.train$Class))

# base model training set MSE values for the BostonHousing data
bh.train.mse.tree.sx <- sapply(bh.bm.tree.sx,
                               function(h) mse(predict(h, bh.train), bh.train$medv))
bh.train.mse.lm.sx <- sapply(bh.bm.lm.sx,
                             function(h) mse(predict(h, bh.train), bh.train$medv))

# base model test set errors for the HouseVotes84 data
hv.test.err.tree.sx <- sapply(hv.bm.tree.sx,
                              function(h) err(predict(h, hv.test, type="c"),
                                              hv.test$Class))
hv.test.err.nb.sx <- sapply(hv.bm.nb.sx,
                            function(h) err(predict(h, hv.test), hv.test$Class))

# base model test set MSE values for the BostonHousing data
bh.test.mse.tree.sx <- sapply(bh.bm.tree.sx,
                              function(h) mse(predict(h, bh.test), bh.test$medv))
bh.test.mse.lm.sx <- sapply(bh.bm.lm.sx,
                            function(h) mse(predict(h, bh.test), bh.test$medv))
##############
#15-3-2.R
##############
## generate base models by instance weighting
base.ensemble.weight.x <- function(formula, data, m, alg, args=NULL,
                                   weights=runif(nrow(data), min=0.3, max=3),
                                   reweight=function(w, p=NULL)
                                     runif(nrow(data), min=0.3, max=3),
                                   predf=predict)
{
  skip.cond(lapply(1:m,
                   function(i)
                   {
                     if (!is.null(weights))
                     {
                       h <- do.call(alg, c(list(formula, data, weights=weights),
                                           args))
                       pred <- predf(h, data)
                       if (!is.null(weights <<- reweight(weights, pred)))
                         h
                     }
                   }),
            is.null)
}

# base models for the HouseVotes84 data
hv.bm.tree.wx <- base.ensemble.weight.x(Class~., hv.train, 50, rpart)

# base models for the BostonHousing data
bh.bm.tree.wx <- base.ensemble.weight.x(medv~., bh.train, 50, rpart)
bh.bm.lm.wx <- base.ensemble.weight.x(medv~., bh.train, 50, lm)

# base model training set errors for the HouseVotes84 data
hv.train.err.tree.wx <- sapply(hv.bm.tree.wx,
                               function(h) err(predict(h, hv.train, type="c"),
                                               hv.train$Class))

# base model training set MSE values for the BostonHousing data
bh.train.mse.tree.wx <- sapply(bh.bm.tree.wx,
                               function(h) mse(predict(h, bh.train), bh.train$medv))
bh.train.mse.lm.wx <- sapply(bh.bm.lm.wx,
                             function(h) mse(predict(h, bh.train), bh.train$medv))

# base model test set errors for the HouseVotes84 data
hv.test.err.tree.wx <- sapply(hv.bm.tree.wx,
                              function(h) err(predict(h, hv.test, type="c"),
                                              hv.test$Class))

# base model test set MSE values for the BostonHousing data
bh.test.mse.tree.wx <- sapply(bh.bm.tree.wx,
                              function(h) mse(predict(h, bh.test), bh.test$medv))
bh.test.mse.lm.wx <- sapply(bh.bm.lm.wx,
                            function(h) mse(predict(h, bh.test), bh.test$medv))
##############
#15-3-3.R
##############
## generate base models by attribute sampling
base.ensemble.sample.a <- function(formula, data, m, alg, args=NULL,
                                   frac=0, replace=TRUE)
{
  attributes <- x.vars(formula, data)
  target <- y.var(formula)
  ns <- ifelse(clip.val(frac, 0, 1)>0, ceiling(frac*length(attributes)),
               ceiling(sqrt(length(attributes))))
  lapply(1:m, function(i)
  {
    sa <- sample(length(attributes), ns)
    do.call(alg, c(list(make.formula(target, attributes[sa]), data),
                   args))
  })
}

# base models for the HouseVotes84 data
hv.bm.tree.sa <- base.ensemble.sample.a(Class~., hv.train, 50, rpart,
                                        args=list(minsplit=2, cp=0))
hv.bm.nb.sa <- base.ensemble.sample.a(Class~., hv.train, 50, naiveBayes)

# base models for the BostonHousing data
bh.bm.tree.sa <- base.ensemble.sample.a(medv~., bh.train, 50, rpart,
                                        args=list(minsplit=2, cp=0))
bh.bm.lm.sa <- base.ensemble.sample.a(medv~., bh.train, 50, lm)

# base model training set errors for the HouseVotes84 data
hv.train.err.tree.sa <- sapply(hv.bm.tree.sa,
                               function(h) err(predict(h, hv.train, type="c"),
                                               hv.train$Class))
hv.train.err.nb.sa <- sapply(hv.bm.nb.sa,
                             function(h) err(predict(h, hv.train), hv.train$Class))

# base model training set MSE values for the BostonHousing data
bh.train.mse.tree.sa <- sapply(bh.bm.tree.sa,
                               function(h) mse(predict(h, bh.train), bh.train$medv))
bh.train.mse.lm.sa <- sapply(bh.bm.lm.sa,
                             function(h) mse(predict(h, bh.train), bh.train$medv))

# base model test set errors for the HouseVotes84 data
hv.test.err.tree.sa <- sapply(hv.bm.tree.sa,
                              function(h) err(predict(h, hv.test, type="c"),
                                              hv.test$Class))
hv.test.err.nb.sa <- sapply(hv.bm.nb.sa,
                            function(h) err(predict(h, hv.test), hv.test$Class))

# base model test set MSE values for the BostonHousing data
bh.test.mse.tree.sa <- sapply(bh.bm.tree.sa,
                              function(h) mse(predict(h, bh.test), bh.test$medv))
bh.test.mse.lm.sa <- sapply(bh.bm.lm.sa,
                            function(h) mse(predict(h, bh.test), bh.test$medv))
##############
#15-3-4.R
##############
## randomized decision tree growing
## with split selection using ns randomly chosen attributes at each node
## (if unspecified or 0, it defaults to the square root of the number of attributes)
grow.randdectree <- function(formula, data, ns=0,
                             imp=entropy.p, maxprob=0.999, minsplit=2, maxdepth=8)
{
  init <- function()
  {
    clabs <<- factor(levels(data[[class]]),
                     levels=levels(data[[class]]))  # class labels
    tree <<- data.frame(node=1, attribute=NA, value=NA, class=NA, count=NA,
                        `names<-`(rep(list(NA), length(clabs)),
                                  paste("p", clabs, sep=".")))
    cprobs <<- (ncol(tree)-length(clabs)+1):ncol(tree)  # class probability columns
    nodemap <<- rep(1, nrow(data))
    n <<- 1
  }
  
  next.node <- function(node)
  {
    if (any(opn <- tree$node>node))
      min(tree$node[opn])
    else Inf
  }
  
  class.distribution <- function(node)
  {
    tree[tree$node==node,"count"] <<- sum(nodemap==node)
    tree[tree$node==node,cprobs] <<- pdisc(data[nodemap==node,class])
  }
  
  class.label <- function(node)
  {
    tree$class[tree$node==node] <<- which.max(tree[tree$node==node,cprobs])
  }
  
  stop.criteria <- function(node)
  {
    node>=2^maxdepth || tree[tree$node==node,"count"]<minsplit ||
      max(tree[tree$node==node,cprobs])>maxprob
  }
  
  split.eval <- function(av, sv, cl)
  {
    cond <- !is.na(av) & (if (is.numeric(av)) av<=as.numeric(sv) else av==sv)
    
    pd1 <- pdisc(cl[cond])
    n1 <- sum(cond)
    pd0 <- pdisc(cl[!cond])
    n0 <- sum(!cond)
    
    if (n1>0 && n0>0)
      weighted.impurity(pd1, n1, pd0, n0, imp)
    else
      Inf
  }
  
  split.select <- function(node)
  {
    splits <- data.frame()
    for (attribute in sample(attributes, ns))
    {
      uav <- sort(unique(data[nodemap==node,attribute]))
      if (length(uav)>1)
        splits <- rbind(splits,
                        data.frame(attribute=attribute,
                                   value=if (is.numeric(uav))
                                     midbrk(uav)
                                   else as.character(uav),
                                   stringsAsFactors=F))
    }
    
    if (nrow(splits)>0)
      splits$eval <- sapply(1:nrow(splits),
                            function(s)
                              split.eval(data[nodemap==node,splits$attribute[s]],
                                         splits$value[s],
                                         data[nodemap==node,class]))
    if ((best.eval <- min(splits$eval))<Inf)
      tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
    return(best.eval)
  }
  
  split.apply <- function(node)
  {
    tree <<- rbind(tree,
                   data.frame(node=(2*node):(2*node+1),
                              attribute=NA, value=NA, class=NA, count=NA,
                              `names<-`(rep(list(NA), length(clabs)),
                                        paste("p", clabs, sep="."))))
    
    av <- data[[tree$attribute[tree$node==node]]]
    cond <- !is.na(av) & (if (is.numeric(av))
      av<=as.numeric(tree$value[tree$node==node])
      else av==tree$value[tree$node==node])
    nodemap[nodemap==node & cond] <<- 2*node
    nodemap[nodemap==node & !cond] <<- 2*node+1
  }
  
  tree <- nodemap <- node <- NULL
  clabs <- cprobs <- NULL
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  ns <- ifelse(ns==0, round(sqrt(length(attributes))),
               clip.val(ns, 1, length(attributes)))
  
  init()
  while (is.finite(node))
  {
    class.distribution(node)
    class.label(node)
    if (!stop.criteria(node))
      if (split.select(node)<Inf)
        split.apply(node)
    node <- next.node(node)
  }
  tree$class <- clabs[tree$class]
  `class<-`(tree, "dectree")
}


## randomized regression tree growing
## with split selection using ns randomly chosen attributes at each node
## (if unspecified or 0, it defaults to the square root of the number of attributes)
grow.randregtree <- function(formula, data, ns=0,
                             minvar=0.005, minsplit=2, maxdepth=8)
{
  init <- function()
  {
    tree <<- data.frame(node=1, attribute=NA, value=NA, target=NA,
                        count=NA, mean=NA, variance=NA)
    nodemap <<- rep(1, nrow(data))
    node <<- 1
  }
  
  next.node <- function(node)
  {
    if (any(opn <- tree$node>node))
      min(tree$node[opn])
    else Inf
  }
  
  target.summary <- function(node)
  {
    tree$count[tree$node==node] <<- sum(nodemap==node)
    tree$mean[tree$node==node] <<- mean(data[nodemap==node,target])
    tree$variance[tree$node==node] <<- var1(data[nodemap==node,target])
  }
  
  target.value <- function(node)
  {
    tree$target[tree$node==node] <<- tree$mean[tree$node==node]
  }
  
  stop.criteria <- function(node)
  {
    node>=2^maxdepth || tree$count[tree$node==node]<minsplit ||
      tree$variance[tree$node==node]<minvar
  }
  
  split.eval <- function(av, sv, tv)
  {
    cond <- !is.na(av) & (if (is.numeric(av)) av<=as.numeric(sv) else av==sv)
    v1 <- tv[cond]
    n1 <- sum(cond)
    v0 <- tv[!cond]
    n0 <- sum(!cond)
    if (n1>0 && n0>0)
      weighted.dispersion(v1, v0)
    else
      Inf
  }
  
  split.select <- function(node)
  {
    splits <- data.frame()
    for (attribute in sample(attributes, ns))
    {
      uav <- sort(unique(data[nodemap==node,attribute]))
      if (length(uav)>1)
        splits <- rbind(splits,
                        data.frame(attribute=attribute,
                                   value=if (is.numeric(uav))
                                     midbrk(uav)
                                   else as.character(uav),
                                   stringsAsFactors=F))
    }
    
    if (nrow(splits)>0)
      splits$eval <- sapply(1:nrow(splits),
                            function(s)
                              split.eval(data[nodemap==node,splits$attribute[s]],
                                         splits$value[s],
                                         data[nodemap==node,target]))
    if ((best.eval <- min(splits$eval))<Inf)
      tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
    best.eval
  }
  
  split.apply <- function(node)
  {
    tree <<- rbind(tree,
                   data.frame(node=(2*node):(2*node+1), attribute=NA, value=NA, target=NA,
                              count=NA, mean=NA, variance=NA))
    
    av <- data[[tree$attribute[tree$node==node]]]
    cond <- !is.na(av) & (if (is.numeric(av))
      av<=as.numeric(tree$value[tree$node==node])
      else av==tree$value[tree$node==node])
    nodemap[nodemap==node & cond] <<- 2*node
    nodemap[nodemap==node & !cond] <<- 2*node+1
  }
  
  tree <- nodemap <- node <- NULL
  target <- y.var(formula)
  attributes <- x.vars(formula, data)
  ns <- ifelse(ns==0, round(length(attributes)/3),
               clip.val(ns, 1, length(attributes)))
  
  init()
  while (is.finite(node))
  {
    target.summary(node)
    target.value(node)
    if (!stop.criteria(node))
      if (split.select(node)<Inf)
        split.apply(node)
    node <- next.node(node)
  }
  `class<-`(tree, "regtree")
}


## generate base models by simple multiple algorithm application
base.ensemble.simple <- function(formula, data, m, alg, args=NULL)
{
  lapply(1:m, function(i) do.call(alg, c(list(formula, data), args)))
}

# base models for the HouseVotes84 data
hv.bm.tree.rnd <- base.ensemble.simple(Class~., hv.train, 50, grow.randdectree)

# base models for the BostonHousing data
bh.bm.tree.rnd <- base.ensemble.simple(medv~., bh.train, 50, grow.randregtree,
                                       args=list(minvar=5))

# base model training set errors for the HouseVotes84 data
hv.train.err.tree.rnd <- sapply(hv.bm.tree.rnd,
                                function(h) err(predict(h, hv.train),
                                                hv.train$Class))

# base model training set MSE values for the BostonHousing data
bh.train.mse.tree.rnd <- sapply(bh.bm.tree.rnd,
                                function(h) mse(predict(h, bh.train), bh.train$medv))

# base model test set errors for the HouseVotes84 data
hv.test.err.tree.rnd <- sapply(hv.bm.tree.rnd,
                               function(h) err(predict(h, hv.test), hv.test$Class))

# base model test set MSE values for the BostonHousing data
bh.test.mse.tree.rnd <- sapply(bh.bm.tree.rnd,
                               function(h) mse(predict(h, bh.test), bh.test$medv))
##############
#15-3-5.R
##############
# base model training set errors for the HouseVotes84 data
boxplot(list(tree.sx=hv.train.err.tree.sx,
             tree.wx=hv.train.err.tree.wx,
             tree.sa=hv.train.err.tree.sa,
             tree.rnd=hv.train.err.tree.rnd,
             nb.sx=hv.train.err.nb.sx,
             nb.sa=hv.train.err.nb.sa),
        main="HouseVotes84 (train)", las=2, col="grey", ylim=c(0, 0.26))

# base model test set errors for the HouseVotes84 data
boxplot(list(tree.sx=hv.test.err.tree.sx,
             tree.wx=hv.test.err.tree.wx,
             tree.sa=hv.test.err.tree.sa,
             tree.rnd=hv.test.err.tree.rnd,
             nb.sx=hv.test.err.nb.sx,
             nb.sa=hv.test.err.nb.sa),
        main="HouseVotes84 (test)", las=2, col="grey", ylim=c(0, 0.26))

# base model training set MSE values for the BostonHousing data
boxplot(list(tree.sx=bh.train.mse.tree.sx,
             tree.wx=bh.train.mse.tree.wx,
             tree.sa=bh.train.mse.tree.sa,
             tree.rnd=bh.train.mse.tree.rnd,
             lm.sx=bh.train.mse.lm.sx,
             lm.wx=bh.train.mse.lm.wx,
             lm.sa=bh.train.mse.lm.sa),
        main="BostonHousing (train)", las=2, col="grey", ylim=c(0, 130))

# base model test set MSE values for the BostonHousing data
boxplot(list(tree.sx=bh.test.mse.tree.sx,
             tree.wx=bh.test.mse.tree.wx,
             tree.sa=bh.test.mse.tree.sa,
             tree.rnd=bh.test.mse.tree.rnd,
             lm.sx=bh.test.mse.lm.sx,
             lm.wx=bh.train.mse.lm.wx,
             lm.sa=bh.test.mse.lm.sa),
        main="BostonHousing (test)", las=2, col="grey", ylim=c(0, 130))
##############
#15-4-1.R
##############
## combine base models by voting/averaging
predict.ensemble.basic <- function(models, data, predf=predict)
{
  bp <- data.frame(lapply(models, function(h) predf(h, data)))
  combf <- if (is.numeric(bp[,1])) mean else modal  # combination scheme
  cp <- sapply(1:nrow(bp), function(i) combf(as.vector(as.matrix(bp[i,]))))
  if (is.factor(bp[,1]))
    factor(cp, levels=levels(bp[,1]))
  else
    cp
}

# combine base models for the HouseVotes84 data
hv.pred.tree.sx.b <- predict.ensemble.basic(hv.bm.tree.sx, hv.test,
                                            predf=function(...)
                                              predict(..., type="c"))
hv.pred.nb.sx.b <- predict.ensemble.basic(hv.bm.nb.sx, hv.test)
hv.pred.tree.wx.b <- predict.ensemble.basic(hv.bm.tree.wx, hv.test,
                                            predf=function(...)
                                              predict(..., type="c"))
hv.pred.tree.sa.b <- predict.ensemble.basic(hv.bm.tree.sa, hv.test,
                                            predf=function(...)
                                              predict(..., type="c"))
hv.pred.nb.sa.b <- predict.ensemble.basic(hv.bm.nb.sa, hv.test)
hv.pred.tree.rnd.b <- predict.ensemble.basic(hv.bm.tree.rnd, hv.test)

# combine base models for the BostonHousing data
bh.pred.tree.sx.b <- predict.ensemble.basic(bh.bm.tree.sx, bh.test)
bh.pred.lm.sx.b <- predict.ensemble.basic(bh.bm.lm.sx, bh.test)
bh.pred.tree.wx.b <- predict.ensemble.basic(bh.bm.tree.wx, bh.test)
bh.pred.lm.wx.b <- predict.ensemble.basic(bh.bm.lm.wx, bh.test)
bh.pred.tree.sa.b <- predict.ensemble.basic(bh.bm.tree.sa, bh.test)
bh.pred.lm.sa.b <- predict.ensemble.basic(bh.bm.lm.sa, bh.test)
bh.pred.tree.rnd.b <- predict.ensemble.basic(bh.bm.tree.rnd, bh.test)

# ensemble model test set errors for the HouseVotes84 data
hv.err.b <- c(tree = hv.err.tree,
              tree.sx = err(hv.pred.tree.sx.b, hv.test$Class),
              tree.wx = err(hv.pred.tree.wx.b, hv.test$Class),
              tree.sa = err(hv.pred.tree.sa.b, hv.test$Class),
              tree.rnd = err(hv.pred.tree.rnd.b, hv.test$Class),
              nb = hv.err.nb,
              nb.sx = err(hv.pred.nb.sx.b, hv.test$Class),
              nb.sa = err(hv.pred.nb.sa.b, hv.test$Class))

# ensemble model test set MSE values for the BostonHousing data
bh.mse.b <- c(tree = bh.mse.tree,
              tree.sx = mse(bh.pred.tree.sx.b, bh.test$medv),
              tree.wx = mse(bh.pred.tree.wx.b, bh.test$medv),
              tree.sa = mse(bh.pred.tree.sa.b, bh.test$medv),
              tree.rnd = mse(bh.pred.tree.rnd.b, bh.test$medv),
              lm = bh.mse.lm,
              lm.sx = mse(bh.pred.lm.sx.b, bh.test$medv),
              lm.wx = mse(bh.pred.lm.wx.b, bh.test$medv),
              lm.sa = mse(bh.pred.lm.sa.b, bh.test$medv))

barplot(hv.err.b, main="HouseVotes84", ylab="Error", las=2)
lines(c(0, 10), rep(hv.err.b[1], 2), lty=2)
lines(c(0, 10), rep(hv.err.b[6], 2), lty=3)

barplot(bh.mse.b, main="Boston Housing", ylab="MSE", las=2)
lines(c(0, 11), rep(bh.mse.b[1], 2), lty=2)
lines(c(0, 11), rep(bh.mse.b[6], 2), lty=3)
##############
#15-4-2.R
##############
## combine base models by probability averaging
predict.ensemble.prob <- function(models, data, predf=predict,
                                  prob=FALSE, labels=NULL)
{
  bp <- lapply(models, function(h) predf(h, data))
  cp <- 0
  for (i in (1:(m <- length(bp))))
    cp <- cp + bp[[i]]
  if (prob)
    cp/m
  else
  {
    if (is.null(labels))
      labels <- colnames(cp)
    factor(apply(cp, 1, which.max), levels=1:2, labels=labels)
  }
}

# combine base models for the HouseVotes84 data
hv.pred.tree.sx.p <- predict.ensemble.prob(hv.bm.tree.sx, hv.test)
hv.pred.nb.sx.p <- predict.ensemble.prob(hv.bm.nb.sx, hv.test,
                                         predf=function(...) predict(..., type="r"))
hv.pred.tree.wx.p <- predict.ensemble.prob(hv.bm.tree.wx, hv.test)
hv.pred.tree.sa.p <- predict.ensemble.prob(hv.bm.tree.sa, hv.test)
hv.pred.nb.sa.p <- predict.ensemble.prob(hv.bm.nb.sa, hv.test,
                                         predf=function(...) predict(..., type="r"))

# ensemble model test set errors for the HouseVotes84 data
hv.err.p <- c(tree = hv.err.tree,
              tree.sx = err(hv.pred.tree.sx.p, hv.test$Class),
              tree.wx = err(hv.pred.tree.wx.p, hv.test$Class),
              tree.sa = err(hv.pred.tree.sa.p, hv.test$Class),
              nb = hv.err.nb,
              nb.sx = err(hv.pred.nb.sx.p, hv.test$Class),
              nb.sa = err(hv.pred.nb.sa.p, hv.test$Class))

barplot(hv.err.p, main="HouseVotes84", ylab="Error", las=2)
lines(c(0, 9), rep(hv.err.p[1], 2), lty=2)
lines(c(0, 9), rep(hv.err.p[5], 2), lty=3)
##############
#15-4-3.R
##############
## combine base models by weighted voting/averaging/summing
predict.ensemble.weighted <- function(models, weights, data, predf=predict,
                                      summing=FALSE)
{
  bp <- data.frame(lapply(models, function(h) predf(h, data)))
  combf <- if (is.numeric(bp[,1])) weighted.mean
  else weighted.modal  # combination scheme
  cp <- sapply(1:nrow(bp), function(i) combf(as.vector(as.matrix(bp[i,])), weights))
  if (is.numeric(bp[,1]) && summing)
    cp <- cp*sum(weights)  # summing instead of averaging requested
  if (is.factor(bp[,1]))
    factor(cp, levels=levels(bp[,1]))
  else
    cp
}

# combine base models for the HouseVotes84 data
hv.pred.tree.sx.w <- predict.ensemble.weighted(hv.bm.tree.sx,
                                               1/(hv.train.err.tree.sx+0.01),
                                               hv.test,
                                               predf=function(...)
                                                 predict(..., type="c"))
hv.pred.nb.sx.w <- predict.ensemble.weighted(hv.bm.nb.sx,
                                             1/(hv.train.err.nb.sx+0.01),
                                             hv.test)
hv.pred.tree.wx.w <- predict.ensemble.weighted(hv.bm.tree.wx,
                                               1/(hv.train.err.tree.wx+0.01),
                                               hv.test,
                                               predf=function(...)
                                                 predict(..., type="c"))
hv.pred.tree.sa.w <- predict.ensemble.weighted(hv.bm.tree.sa,
                                               1/(hv.train.err.tree.sa+0.01),
                                               hv.test,
                                               predf=function(...)
                                                 predict(..., type="c"))
hv.pred.nb.sa.w <- predict.ensemble.weighted(hv.bm.nb.sa,
                                             1/(hv.train.err.nb.sa+0.01), hv.test)
hv.pred.tree.rnd.w <- predict.ensemble.weighted(hv.bm.tree.rnd,
                                                1/(hv.train.err.tree.rnd+0.01),
                                                hv.test)

# combine base models for the BostonHousing data
bh.pred.tree.sx.w <- predict.ensemble.weighted(bh.bm.tree.sx,
                                               1/(bh.train.mse.tree.sx+1), bh.test)
bh.pred.lm.sx.w <- predict.ensemble.weighted(bh.bm.lm.sx,
                                             1/(bh.train.mse.lm.sx+1), bh.test)
bh.pred.tree.wx.w <- predict.ensemble.weighted(bh.bm.tree.wx,
                                               1/(bh.train.mse.tree.wx+1), bh.test)
bh.pred.lm.wx.w <- predict.ensemble.weighted(bh.bm.lm.wx,
                                             1/(bh.train.mse.lm.wx+1), bh.test)
bh.pred.tree.sa.w <- predict.ensemble.weighted(bh.bm.tree.sa,
                                               1/(bh.train.mse.tree.sa+1), bh.test)
bh.pred.lm.sa.w <- predict.ensemble.weighted(bh.bm.lm.sa,
                                             1/(bh.train.mse.lm.sa+1), bh.test)
bh.pred.tree.rnd.w <- predict.ensemble.weighted(bh.bm.tree.rnd,
                                                1/(bh.train.mse.tree.rnd+1), bh.test)

# ensemble model test set errors for the HouseVotes84 data
hv.err.w <- c(tree = hv.err.tree,
              tree.sx = err(hv.pred.tree.sx.w, hv.test$Class),
              tree.wx = err(hv.pred.tree.wx.w, hv.test$Class),
              tree.sa = err(hv.pred.tree.sa.w, hv.test$Class),
              tree.rnd = err(hv.pred.tree.rnd.w, hv.test$Class),
              nb = hv.err.nb,
              nb.sx = err(hv.pred.nb.sx.w, hv.test$Class),
              nb.sa = err(hv.pred.nb.sa.w, hv.test$Class))

# ensemble model test set MSE values for the BostonHousing data
bh.mse.w <- c(tree = bh.mse.tree,
              tree.sx = mse(bh.pred.tree.sx.w, bh.test$medv),
              tree.wx = mse(bh.pred.tree.wx.w, bh.test$medv),
              tree.sa = mse(bh.pred.tree.sa.w, bh.test$medv),
              tree.rnd = mse(bh.pred.tree.rnd.w, bh.test$medv),
              lm = bh.mse.lm,
              lm.sx = mse(bh.pred.lm.sx.w, bh.test$medv),
              lm.wx = mse(bh.pred.lm.wx.w, bh.test$medv),
              lm.sa = mse(bh.pred.lm.sa.w, bh.test$medv))

barplot(hv.err.w, main="HouseVotes84", ylab="Error", las=2)
lines(c(0, 10), rep(hv.err.w[1], 2), lty=2)
lines(c(0, 10), rep(hv.err.w[6], 2), lty=3)

barplot(bh.mse.w, main="Boston Housing", ylab="MSE", las=2)
lines(c(0, 11), rep(bh.mse.w[1], 2), lty=2)
lines(c(0, 11), rep(bh.mse.w[6], 2), lty=3)
##############
#15-4-4.R
##############
## combine base models by using as attributes
## create a model using the specified algorithm and training data
combine.ensemble.attributes <- function(models, data, alg, args=NULL, predf=predict,
                                        append=FALSE)
{
  target <- as.character(models[[1]]$terms[[2]])
  tind <- match(target, names(data))
  data.base <- `names<-`(cbind(data.frame(lapply(models,
                                                 function(h) predf(h, data))),
                               data[[target]]),
                         c(paste("h", 1:length(models), sep=""), target))
  if (append)
    data.base <- cbind(data[,-tind], data.base)
  do.call(alg, c(list(make.formula(target, "."), data.base), args))
}

## combine base models by using as attributes
## predict using the specified base models and combined model
predict.ensemble.attributes <- function(combined.model, base.models, data,
                                        combined.predf=predict, base.predf=predict)
{
  data.pred <- `names<-`(data.frame(lapply(base.models,
                                           function(h) base.predf(h, data))),
                         paste("h", 1:length(base.models), sep=""))
  data.pred <- cbind(data, data.pred)  # make the original attributes available
  combined.predf(combined.model, data.pred)
}

# combine base models for the HouseVotes84 data
hv.tree.sx.nb <- combine.ensemble.attributes(hv.bm.tree.sx, hv.train, naiveBayes,
                                             predf=function(...)
                                               predict(..., type="c"))
hv.pred.tree.sx.nb <- predict.ensemble.attributes(hv.tree.sx.nb, hv.bm.tree.sx,
                                                  hv.test,
                                                  base.predf=function(...)
                                                    predict(..., type="c"))
hv.tree.wx.nb <- combine.ensemble.attributes(hv.bm.tree.wx, hv.train, naiveBayes,
                                             predf=function(...)
                                               predict(..., type="c"))
hv.pred.tree.wx.nb <- predict.ensemble.attributes(hv.tree.wx.nb, hv.bm.tree.wx,
                                                  hv.test,
                                                  base.predf=function(...)
                                                    predict(..., type="c"))
hv.tree.sa.nb <- combine.ensemble.attributes(hv.bm.tree.sa, hv.train, naiveBayes,
                                             predf=function(...)
                                               predict(..., type="c"))
hv.pred.tree.sa.nb <- predict.ensemble.attributes(hv.tree.sa.nb, hv.bm.tree.sa,
                                                  hv.test,
                                                  base.predf=function(...)
                                                    predict(..., type="c"))

# combine base models for the BostonHousing data
bh.tree.sx.lm <- combine.ensemble.attributes(bh.bm.tree.sx, bh.train, lm)
bh.pred.tree.sx.lm <- predict.ensemble.attributes(bh.tree.sx.lm, bh.bm.tree.sx,
                                                  bh.test)
bh.tree.wx.lm <- combine.ensemble.attributes(bh.bm.tree.wx, bh.train, lm)
bh.pred.tree.wx.lm <- predict.ensemble.attributes(bh.tree.wx.lm, bh.bm.tree.wx,
                                                  bh.test)
bh.tree.sa.lm <- combine.ensemble.attributes(bh.bm.tree.sa, bh.train, lm)
bh.pred.tree.sa.lm <- predict.ensemble.attributes(bh.tree.sa.lm, bh.bm.tree.sa,
                                                  bh.test)

# ensemble model test set errors for the HouseVotes84 data
hv.err.a <- c(tree = hv.err.tree,
              tree.sx.nb = err(hv.pred.tree.sx.nb, hv.test$Class),
              tree.wx.nb = err(hv.pred.tree.wx.nb, hv.test$Class),
              tree.sa.nb = err(hv.pred.tree.sa.nb, hv.test$Class))

# ensemble model test set MSE values for the BostonHousing data
bh.mse.a <- c(tree = bh.mse.tree,
              tree.sx.lm = mse(bh.pred.tree.sx.lm, bh.test$medv),
              tree.wx.lm = mse(bh.pred.tree.wx.lm, bh.test$medv),
              tree.sa.lm = mse(bh.pred.tree.sa.lm, bh.test$medv))

barplot(hv.err.a, main="HouseVotes84", ylab="Error", las=2)
lines(c(0, 5), rep(hv.err.a[1], 2), lty=2)

barplot(bh.mse.a, main="Boston Housing", ylab="MSE", las=2)
lines(c(0, 5), rep(bh.mse.a[1], 2), lty=2)
##############
#15-5-1.R
##############
## bagging ensemble modeling using m base models created with algorithm alg
## with arguments arg
bagging <- function(formula, data, m, alg, args=NULL)
{
  `class<-`(base.ensemble.sample.x(formula, data, m, alg, args), "bagging")
}

## bagging prediction
predict.bagging <- function(models, data, predf=predict)
{
  predict.ensemble.basic(models, data, predf)
}

# bagging for the HouseVotes84 data
hv.bagg.tree <- bagging(Class~., hv.train, 50, rpart, args=list(minsplit=2, cp=0))
hv.bagg.nb <- bagging(Class~., hv.train, 50, naiveBayes)

hv.pred.bagg.tree <- predict(hv.bagg.tree, hv.test,
                             predf=function(...) predict(..., type="c"))
hv.pred.bagg.nb <- predict(hv.bagg.nb, hv.test)

# bagging for the BostonHousing data
bh.bagg.tree <- bagging(medv~., bh.train, 50, rpart, args=list(minsplit=2, cp=0))
bh.bagg.lm <- bagging(medv~., bh.train, 50, lm)

bh.pred.bagg.tree <- predict(bh.bagg.tree, bh.test)
bh.pred.bagg.lm <- predict(bh.bagg.lm, bh.test)

# bagging test set errors for the HouseVotes84 data
hv.err.bagg <- list(tree = err(hv.pred.bagg.tree, hv.test$Class),
                    nb = err(hv.pred.bagg.nb, hv.test$Class))

# bagging test set MSE values for the BostonHousing data
bh.mse.bagg <- list(tree = mse(bh.pred.bagg.tree, bh.test$medv),
                    lm = mse(bh.pred.bagg.lm, bh.test$medv))
##############
#15-5-2.R
##############
curve(0.5*log((1-x)/x), from=0, to=0.5,
      xlab="model error", ylab="model weight")
curve(exp(0.5*log((1-x)/x)), from=0, to=0.5,
      xlab="model error", ylab="instance weight multiplier", ylim=c(0, 10), lty=2)
curve(exp(-0.5*log((1-x)/x)), from=0, to=0.5, lty=3, add=TRUE)
legend("topright", legend=c("misclassified", "correctly classified"), lty=2:3)
##############
#15-5-3.R
##############
## AdaBoost ensemble modeling using up to m base models created using algorithm alg
## with arguments arg and maximum allowed base model error 0.5-eps
adaboost <- function(formula, data, m, alg, eps=0.01, args=NULL, predf=predict)
{
  class <- y.var(formula)
  nc <- nlevels(data[[class]])
  model.weights <- NULL
  
  abst.reweight <- function(weights, pred)
  {
    e <- werr(pred, data[[class]], weights)
    if (e<=0.5-eps && is.finite(mw <- 0.5*(log((1-e)/e)+log(nc-1))))
    {
      model.weights <<- c(model.weights, mw)
      weights*exp(mw*(2*(pred!=data[[class]])-1))
    }
    else
      NULL
  }
  
  `class<-`(list(models=base.ensemble.weight.x(formula, data, m, alg, args,
                                               weights=rep(1, nrow(data)),
                                               abst.reweight,
                                               predf=predf),
                 model.weights=model.weights), "adaboost")
}

## AdaBoost prediction
predict.adaboost <- function(boost, data, predf=predict)
{
  predict.ensemble.weighted(boost$models, boost$model.weights, data, predf)
}

# AdaBoost for the HouseVotes84 data
hv.abst.tree1 <- adaboost(Class~., hv.train, 50, rpart,
                          args=list(minsplit=2, cp=0, maxdepth=1),
                          predf=function(...) predict(..., type="c"))
hv.abst.tree3 <- adaboost(Class~., hv.train, 50, rpart,
                          args=list(minsplit=2, cp=0, maxdepth=3),
                          predf=function(...) predict(..., type="c"))
hv.abst.tree5 <- adaboost(Class~., hv.train, 50, rpart,
                          args=list(minsplit=2, cp=0, maxdepth=5),
                          predf=function(...) predict(..., type="c"))

hv.pred.abst.tree1 <- predict(hv.abst.tree1, hv.test,
                              predf=function(...) predict(..., type="c"))
hv.pred.abst.tree3 <- predict(hv.abst.tree3, hv.test,
                              predf=function(...) predict(..., type="c"))
hv.pred.abst.tree5 <- predict(hv.abst.tree5, hv.test,
                              predf=function(...) predict(..., type="c"))

# AdaBoost test set errors for the HouseVotes84 data
hv.err.abst <- list(tree1 = err(hv.pred.abst.tree1, hv.test$Class),
                    tree3 = err(hv.pred.abst.tree3, hv.test$Class),
                    tree5 = err(hv.pred.abst.tree5, hv.test$Class))
##############
#15-5-4.R
##############
## gradient boosting ensemble modeling using up to m base models
## created with algorithm alg with arguments arg
gradboost <- function(formula, data, m, alg, beta=0.1, args=NULL, predf=predict)
{
  attributes <- x.vars(formula, data)
  aind <- match(attributes, names(data))
  f <- y.var(formula)
  find <- match(f, names(data))
  
  models <- list(do.call(alg, c(list(formula, data), args)))
  model.weights <- 1
  
  for (i in (2:m))
  {
    res <- data[,find]-predict.gradboost(list(models=models,
                                              model.weights=model.weights),
                                         data, predf=predf)
    data.i <- eval(parse(text=paste("cbind(data[,aind],", f, "=res)")))
    models <- c(models, list(h <- do.call(alg, c(list(formula, data.i), args))))
    model.weights <- c(model.weights,
                       beta*sum(res*(pred <- predf(h, data)))/sum(pred^2))
  }
  `class<-`(list(models=models, model.weights=model.weights), "gradboost")
}

## gradient boosting prediction
predict.gradboost <- function(boost, data, predf=predict)
{
  predict.ensemble.weighted(boost$models, boost$model.weights, data, predf,
                            summing=TRUE)
}

# gradient boosting for the BostonHousing data
bh.gbst.tree1 <- gradboost(medv~., bh.train, 50, rpart,
                           args=list(minsplit=2, cp=0, maxdepth=1))
bh.gbst.tree3 <- gradboost(medv~., bh.train, 50, rpart,
                           args=list(minsplit=2, cp=0, maxdepth=3))
bh.gbst.tree5 <- gradboost(medv~., bh.train, 50, rpart,
                           args=list(minsplit=2, cp=0, maxdepth=5))
bh.gbst.lm <- gradboost(medv~., bh.train, 50, lm)

bh.pred.gbst.tree1 <- predict(bh.gbst.tree1, bh.test)
bh.pred.gbst.tree3 <- predict(bh.gbst.tree3, bh.test)
bh.pred.gbst.tree5 <- predict(bh.gbst.tree5, bh.test)
bh.pred.gbst.lm <- predict(bh.gbst.lm, bh.test)

# gradient boosting test set MSE values for the BostonHousing data
bh.mse.gbst <- list(tree1 = mse(bh.pred.gbst.tree1, bh.test$medv),
                    tree3 = mse(bh.pred.gbst.tree3, bh.test$medv),
                    tree5 = mse(bh.pred.gbst.tree5, bh.test$medv),
                    lm = mse(bh.pred.gbst.lm, bh.test$medv))
##############
#15-5-5.R
##############
## random forest ensemble modeling using m randomized decision or regression trees
## with ns randomly selected attributes at each node used for splitting
randforest <- function(formula, data, m, ns=0, args=NULL)
{
  target <- y.var(formula)
  alg <- if (!is.numeric(data[[target]])) grow.randdectree else grow.randregtree
  
  `class<-`(base.ensemble.sample.x(formula, data, m, alg, args=c(list(ns=ns), args)),
            "randforest")
}

## random forest prediction
predict.randforest <- function(rf, data)
{
  predict.ensemble.basic(rf, data)
}

# random forest for the HouseVotes84 data
hv.rf.tree3 <- randforest(Class~., hv.train, 50, args=list(maxdepth=3))
hv.rf.tree5 <- randforest(Class~., hv.train, 50, args=list(maxdepth=5))
hv.rf.tree8 <- randforest(Class~., hv.train, 50, args=list(maxdepth=8))

hv.pred.rf.tree3 <- predict(hv.rf.tree3, hv.test)
hv.pred.rf.tree5 <- predict(hv.rf.tree5, hv.test)
hv.pred.rf.tree8 <- predict(hv.rf.tree8, hv.test)

# random forest for the BostonHousing data
bh.rf.tree3 <- randforest(medv~., bh.train, 50, args=list(maxdepth=3))
bh.rf.tree5 <- randforest(medv~., bh.train, 50, args=list(maxdepth=5))
bh.rf.tree8 <- randforest(medv~., bh.train, 50, args=list(maxdepth=8))

bh.pred.rf.tree3 <- predict(bh.rf.tree3, bh.test)
bh.pred.rf.tree5 <- predict(bh.rf.tree8, bh.test)
bh.pred.rf.tree8 <- predict(bh.rf.tree8, bh.test)

# random forest test set errors for the HouseVotes84 data
hv.err.rf <- list(tree3 = err(hv.pred.rf.tree3, hv.test$Class),
                  tree5 = err(hv.pred.rf.tree5, hv.test$Class),
                  tree8 = err(hv.pred.rf.tree8, hv.test$Class))

# random forest test set MSE values for the BostonHousing data
bh.mse.rf <- list(tree3 = mse(bh.pred.rf.tree3, bh.test$medv),
                  tree5 = mse(bh.pred.rf.tree5, bh.test$medv),
                  tree8 = mse(bh.pred.rf.tree8, bh.test$medv))
##############
#15-5-6.R
##############
## random naive Bayes ensemble modeling using m base models
## each with ns randomly selected attributes
## (if unspecified, it defaults to the square root of the number of attributes)
randnaiveBayes <- function(formula, data, m, ns=0)
{
  attributes <- x.vars(formula, data)
  target <- y.var(formula)
  ns <- ifelse(ns==0, round(sqrt(length(attributes))),
               clip.val(ns, 1, length(attributes)))
  
  `class<-`(lapply(1:m, function(i)
  {
    bag <- sample(nrow(data), size=nrow(data), replace=TRUE)
    sa <- sample(length(attributes), ns)
    naiveBayes(make.formula(target, attributes[sa]),
               data[bag,])
  }), "randnaiveBayes")
}

## random naive Bayes prediction
predict.randnaiveBayes <- function(rnb, data, prob=FALSE)
{
  predict.ensemble.prob(rnb, data, predf=function(...) predict(..., type="r"),
                        prob=prob, labels=rnb[[1]]$levels)
}

# random naive Bayes for the HouseVotes84 data
hv.rnb <- randnaiveBayes(Class~., hv.train, 500)
hv.pred.rnb <- predict(hv.rnb, hv.test)
# random naive Bayes test set error for the HouseVotes84 data
hv.err.rnb <- list(nb = err(hv.pred.rnb, hv.test$Class))
##############
#15-6-1.R
##############
hv.err <- c(tree=hv.err.tree,
            bagg.tree=hv.err.bagg$tree,
            abst.tree1=hv.err.abst$tree1,
            abst.tree3=hv.err.abst$tree3,
            abst.tree5=hv.err.abst$tree5,
            rf.tree3=hv.err.rf$tree3,
            rf.tree5=hv.err.rf$tree5,
            rf.tree8=hv.err.rf$tree8,
            nb=hv.err.nb,
            bagg.nb=hv.err.bagg$nb,
            rnb=hv.err.rnb$nb)

bh.mse <- c(tree=bh.mse.tree,
            bagg.tree=bh.mse.bagg$tree,
            gbst.tree1=bh.mse.gbst$tree1,
            gbst.tree3=bh.mse.gbst$tree3,
            gbst.tree5=bh.mse.gbst$tree5,
            rf.tree3=bh.mse.rf$tree3,
            rf.tree5=bh.mse.rf$tree5,
            rf.tree8=bh.mse.rf$tree8,
            lm=bh.mse.lm,
            bagg.lm=bh.mse.bagg$lm,
            gbst.lm=bh.mse.gbst$lm)

barplot(hv.err, main="HouseVotes84", ylab="Error", las=2)
lines(c(0, 13), rep(hv.err[1], 2), lty=2)
lines(c(0, 13), rep(hv.err[9], 2), lty=3)

barplot(bh.mse, main="Boston Housing", ylab="MSE", las=2)
lines(c(0, 13), rep(bh.mse[1], 2), lty=2)
lines(c(0, 13), rep(bh.mse[9], 2), lty=3)