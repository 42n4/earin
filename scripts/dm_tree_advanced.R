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

mdata<-HouseVotes84

rhv <- runif(nrow())
hv.train <- mdata[rhv>=0.33,]
hv.test <- mdata[rhv<0.33,]

## a simple decision tree growing implementation
## with missing value support using fractional instances
grow.dectree.frac <- function(formula, data,
                              imp=entropy.p, maxprob=0.999, minsplit=2, maxdepth=8)
{
  nmn <- function(node) { nodemap[,"node"]==node }            # nodemap entries for node node
  inn <- function(node)
  { nodemap[nodemap[,"node"]==node,"instance"] }                  # instances at node node
  wgn <- function(node) { nodemap[nodemap[,"node"]==node,"weight"] }   # weights at node node
  
  init <- function()
  {
    clabs <<- factor(levels(data[[class]]),
                     levels=levels(data[[class]]))      # class labels
    tree <<- data.frame(node=1, attribute=NA, value=NA, class=NA, count=NA,
                        `names<-`(rep(list(NA), length(clabs)),
                                  paste("p", clabs, sep=".")))
    cprobs <<- (ncol(tree)-length(clabs)+1):ncol(tree)  # class probability columns
    nodemap <<- cbind(instance=1:nrow(data), node=rep(1, nrow(data)),
                      weight=rep(1, nrow(data)))
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
    tree[tree$node==node,"count"] <<- sum(wgn(node))
    tree[tree$node==node,cprobs] <<- weighted.pdisc(data[inn(node),class], w=wgn(node))
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
  
  split.eval <- function(av, sv, cl, w)
  {
    cond <- if (is.numeric(av)) av<=as.numeric(sv) else av==sv
    cond1 <- !is.na(av) & cond   # true split outcome
    cond0 <- !is.na(av) & !cond  # false split outcome
    
    pd1 <- weighted.pdisc(cl[cond1], w=w[cond1])
    n1 <- sum(w[cond1])
    pd0 <- weighted.pdisc(cl[cond0], w=w[cond0])
    n0 <- sum(w[cond0])
    pdm <- weighted.pdisc(cl[is.na(av)], w=w[is.na(av)])
    nm <- sum(w[is.na(av)])
    
    if (nm>0)
    {
      p1 <- if (n1+n0>0) n1/(n1+n0) else 0.5
      p0 <- 1-p1
      pd1 <- (n1*pd1 + p1*nm*pdm)/(n1+p1*nm)
      n1 <- n1 + p1*nm
      pd0 <- (n0*pd0 + p0*nm*pdm)/(n0+p0*nm)
      n0 <- n0 + nm*p0
    }
    
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
      uav <- sort(unique(data[inn(node),attribute]))
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
                              split.eval(data[inn(node),splits$attribute[s]],
                                         splits$value[s],
                                         data[inn(node),class], wgn(node)))
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
    
    av <- data[nodemap[,"instance"],tree$attribute[tree$node==node]]
    cond <- if (is.numeric(av)) av<=as.numeric(tree$value[tree$node==node])
    else av==tree$value[tree$node==node]
    cond1 <- !is.na(av) & cond   # true split outcome
    cond0 <- !is.na(av) & !cond  # false split outcome
    
    n1 <- sum(nodemap[nmn(node) & cond1,"weight"])
    n0 <- sum(nodemap[nmn(node) & cond0,"weight"])
    nm <- sum(nodemap[nmn(node) & is.na(av),"weight"])
    
    nodemap[nmn(node) & cond1,"node"] <<- 2*node
    nodemap[nmn(node) & cond0,"node"] <<- 2*node+1
    
    if (nm>0)
    {
      p1 <- if (n1+n0>0) n1/(n1+n0) else 0.5
      p0 <- 1-p1
      newnn <- nodemap[nmn(node) & is.na(av),,drop=FALSE]
      nodemap[nmn(node) & is.na(av),"weight"] <<-
        p1*nodemap[nmn(node) & is.na(av),"weight"]
      nodemap[nmn(node) & is.na(av),"node"] <<- 2*node
      newnn[,"weight"] <- p0*newnn[,"weight"]
      newnn[,"node"] <- 2*node+1
      nodemap <<- rbind(nodemap, newnn)
    }
  }
  
  tree <- cprobs <- nodemap <- node <- NULL
  clabs <- cprobs <- NULL
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  
  init()
  while (is.finite(node))
  {
    class.distribution(node)
    class.label(node)
    if (!stop.criteria(node))
      if (split.select(node)<Inf)
        split.apply(node)
    node <- next.node(node)
    tree
    nodemap
    is.finite(node)
  }
  tree$class <- clabs[tree$class]
  `class<-`(tree, "dectree.frac")
}

## convert a dectree.frac object to a data frame
as.data.frame.dectree.frac <- function(x, row.names=NULL, optional=FALSE, ...)
{ as.data.frame(unclass(x), row.names=row.names, optional=optional) }

treem <- grow.dectree.frac(Class~., hv.train)

# data frame conversion
as.data.frame(treem)

##############
#3-7-2.R
##############
## decision tree prediction
## with missing value support using fractional instances
predict.dectree.frac <- function(tree, data)
{
  nmn <- function(node) { nodemap[,"node"]==node }  # nodemap entries for node node
  
  descend <- function(node)
  {
    if (!is.na(tree$attribute[tree$node==node]))  # unless reached a leaf
    {
      av <- data[nodemap[,"instance"],tree$attribute[tree$node==node]]
      cond <- if (is.numeric(av)) av<=as.numeric(tree$value[tree$node==node])
      else av==tree$value[tree$node==node]
      cond1 <- !is.na(av) & cond   # true split outcome
      cond0 <- !is.na(av) & !cond  # false split outcome
      
      nodemap[nmn(node) & cond1, "node"] <<- 2*node
      nodemap[nmn(node) & cond0, "node"] <<- 2*node+1
      
      if (sum(nodemap[nmn(node) & is.na(av), "weight"])>0)
      {
        n1 <- tree$count[tree$node==2*node]
        n0 <- tree$count[tree$node==2*node+1]
        p1 <- if (n1+n0>0) n1/(n1+n0) else 0.5
        p0 <- 1-p1
        
        newnn <- nodemap[nmn(node) & is.na(av),,drop=FALSE]
        nodemap[nmn(node) & is.na(av),"weight"] <<-
          p1*nodemap[nmn(node) & is.na(av),"weight"]
        nodemap[nmn(node) & is.na(av), "node"] <<- 2*node
        newnn[,"weight"] <- p0*newnn[,"weight"]
        newnn[,"node"] <- 2*node+1
        nodemap <<- rbind(nodemap, newnn)
      }
      
      descend(2*node)
      descend(2*node+1)
    }
  }
  
  nodemap <- cbind(instance=1:nrow(data), node=rep(1, nrow(data)),
                   weight=rep(1, nrow(data)))
  descend(1)
  
  clabs <- factor(levels(tree$class), levels=levels(tree$class))
  votes <- merge(nodemap, as.data.frame(tree)[,c("node", "class",
                                                 paste("p", clabs, sep="."))])
  cprobs <- (ncol(votes)-length(clabs)+1):ncol(votes)
  clabs[by(votes, votes$instance,
           function(v) which.max(colSums(v$weight*v[,cprobs])))]
}

# decision tree prediction for the given data with missing attribute values
predict(treem, weatherm)

err(predict(dectree, hv.train), hv.train$Class)

err(predict(dectree, hv.test), hv.test$Class)
