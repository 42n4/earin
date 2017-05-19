library(dmr.claseval)
library(dmr.dectree)
library(dmr.regeval)
library(dmr.regtree)
library(dmr.stats)
library(dmr.util)

library(rpart)
library(e1071)


set.seed(12)

data(HouseVotes84, package="mlbench")
#data(BostonHousing, package="mlbench")
mdata<-HouseVotes84

#data(iris)
#mdata<-iris
rhv <- runif(nrow(mdata))
hv.train <- mdata[rhv>=0.33,]
hv.test <- mdata[rhv<0.33,]


formula<-Class~.
data<-hv.train
imp=entropy.p
maxprob=0.999
minsplit=2
maxdepth=8

grow.dectree <- function(formula, data,
                         imp=entropy.p, maxprob=0.999, minsplit=2, maxdepth=8)
{
  init <- function()
  {
    clabs <<- factor(levels(data[[class]]),
                     levels=levels(data[[class]]))      # class labels
    tree <<- data.frame(node=1, attribute=NA, value=NA, class=NA, count=NA,
                        `names<-`(rep(list(NA), length(clabs)),
                                  paste("p", clabs, sep=".")))
    cprobs <<- (ncol(tree)-length(clabs)+1):ncol(tree)  # class probability columns
    nodemap <<- rep(1, nrow(data))
    node <<- 1
  }
  
  next.node <- function(node)
  {
    if (any(opn <- tree$node>node))
      min(tree$node[opn])
    else Inf
  }
  
  class.distribution <- function(node)
  {
    tree$count[tree$node==node] <<- sum(nodemap==node)
    tree[tree$node==node,cprobs] <<- pdisc(data[nodemap==node,class])
  }
  
  class.label <- function(node)
  {
    tree$class[tree$node==node] <<- which.max(tree[tree$node==node,cprobs])
  }
  
  stop.criteria <- function(node)
  {
    node>=2^maxdepth || tree$count[tree$node==node]<minsplit ||
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
    for (attribute in attributes)
    {
      uav <- sort(unique(data[nodemap==node,attribute]))
      if (length(uav)>1)
        splits <- rbind(splits,
                        data.frame(attribute=attribute,
                                   value=if (is.numeric(uav))
                                     midbrk(uav)
                                   else as.character(uav),
                                   stringsAsFactors=FALSE))
    }
    
    if (nrow(splits)>0)
      splits$eval <- sapply(1:nrow(splits),
                            function(s)
                              split.eval(data[nodemap==node,splits$attribute[s]],
                                         splits$value[s],
                                         data[nodemap==node,class]))
    if ((best.eval <- min(splits$eval))<Inf)
      tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
    best.eval
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
  
  init()
  while (is.finite(node))
  {
    class.distribution(node)
    class.label(node)
    #imp<-entropy.p
    #imp<-gini.p
    #split.select(node)
    #tree
    if (!stop.criteria(node))
      if (split.select(node)<Inf)
        split.apply(node)
    node <- next.node(node)
    tree
    nodemap
    is.finite(node)
  }
  tree$class <- clabs[tree$class]
  `class<-`(tree, "dectree")
}

## convert a dectree object to a data frame
as.data.frame.dectree <- function(x, row.names=NULL, optional=FALSE, ...)
{ as.data.frame(unclass(x), row.names=row.names, optional=optional) }

# grow a decision tree for the HouseVotes84 data
dectree <- grow.dectree(Class~., hv.train)

# data frame conversion
as.data.frame(dectree)



predict(dectree, hv.test)

err(predict(dectree, hv.train), hv.train$Class)

err(predict(dectree, hv.test), hv.test$Class)

