
##############
#20-3.R
##############
library(dmr.regeval)
library(dmr.util)
library(dmr.trans)

library(rpart)
library(rpart.plot)
library(randomForest)

# read column names (extracted from the data description web page)
commnorm.names <- read.table("../Data/commnorm.names",
                             stringsAsFactors=FALSE)[,1]
# read the actual data
commnorm <- read.table("../Data/communities.data",
                       sep=",", na.strings="?", col.names=commnorm.names)
# input attribute names
cn.input.attrs <- names(commnorm)[6:127]

set.seed(12)

rcn <- runif(nrow(commnorm))
cn.train <- commnorm[rcn>=0.33,]
cn.val <- commnorm[rcn<0.33,]

sum(complete.cases(cn.train))/nrow(cn.train)
sum(complete.cases(cn.val))/nrow(cn.val)

# attributes with many (>50%) missing values
cn.input.attrs.miss <-
  names(which(sapply(cn.input.attrs,
                     function(a)
                       sum(is.na(cn.train[a]))/nrow(cn.train))>0.5))

# attributes with many (>10%) outliers
cn.input.attrs.out <-
  names(which(sapply(cn.input.attrs,
                     function(a)
                       length(boxplot(cn.train[a], range=2, plot=FALSE)$out)/
                       nrow(cn.train))>0.1))

cn.input.attrs.cor <- cor(cn.train[,cn.input.attrs], use="pairwise.complete.obs")
cn.input.attrs.corind <- which(upper.tri(cn.input.attrs.cor) &
                                 abs(cn.input.attrs.cor)>0.98, arr.ind=TRUE)
cn.input.attrs.corpairs <- data.frame(a1=cn.input.attrs[cn.input.attrs.corind[,1]],
                                      a2=cn.input.attrs[cn.input.attrs.corind[,2]])

cn.impm <- imp.all(make.formula(NULL, cn.input.attrs), cn.train)
cni.train <- predict.imp(cn.impm, cn.train)
cni.val <- predict.imp(cn.impm, cn.val)

cn.tree.d <- rpart(make.formula("ViolentCrimesPerPop", cn.input.attrs), cn.train)
r2(predict(cn.tree.d, cn.val), cn.val$ViolentCrimesPerPop)

# fully-grown tree
cn.tree.f <- rpart(make.formula("ViolentCrimesPerPop", cn.input.attrs), cn.train,
                   minsplit=2, cp=0)
r2(predict(cn.tree.f, cn.val), cn.val$ViolentCrimesPerPop)
# minimum-error cost-complexity pruning
cn.tree.pmin <- prune(cn.tree.f, cpmin(cn.tree.f$cptable))
r2(predict(cn.tree.pmin, cn.val), cn.val$ViolentCrimesPerPop)
# 1-sd cost-complexity pruning
cn.tree.p1sd <- prune(cn.tree.f, cp1sd(cn.tree.f$cptable))
r2(predict(cn.tree.p1sd, cn.val), cn.val$ViolentCrimesPerPop)

# 10x10-fold cross-validated R2 values for the most promising cp sequence
cn.cp.cv <-
  sapply(unname(cpminrange(cn.tree.f$cptable, 5, 10)),
         function(cp)
         {
           cv <- crossval(rpart, make.formula("ViolentCrimesPerPop", cn.input.attrs),
                          cn.train, args=list(cp=cp, minsplit=2, xval=0), n=10)
           `names<-`(r2(cv$pred, cv$true), cp)
         })

cn.tree.pcv <- prune(cn.tree.f, as.numeric(names(cn.cp.cv)[which.max(cn.cp.cv)]))
r2(predict(cn.tree.pcv, cn.val), cn.val$ViolentCrimesPerPop)

cn.lm <- lm(make.formula("ViolentCrimesPerPop", cn.input.attrs), cni.train)
r2(predict(cn.lm, cni.val), cni.val$ViolentCrimesPerPop)

signif.attrs <- cn.input.attrs[(summary(cn.lm)$coefficients)[-1,4]<0.05]
cn.lm.s <- lm(make.formula("ViolentCrimesPerPop", signif.attrs), cni.train)
r2(predict(cn.lm.s, cni.val), cni.val$ViolentCrimesPerPop)

cn.rf <- randomForest(make.formula("ViolentCrimesPerPop", cn.input.attrs), cni.train,
                      importance=TRUE)
r2(predict(cn.rf, cni.val[,cn.input.attrs]), cni.val$ViolentCrimesPerPop)

varImpPlot(cn.rf, type=1)

cn.attr.utl <- sort(importance(cn.rf)[,1], decreasing=TRUE)
cn.asets <-
  `names<-`(lapply(c(10, 25, 50, 100),
                   function(p)
                     names(cn.attr.utl)[1:round(p*length(cn.attr.utl)/100)]),
            paste("as", c(10, 25, 50, 100), "p", sep=""))

cn.attr.cor <- sort(abs(cor(cn.train[,cn.input.attrs], cn.train$ViolentCrimesPerPop,
                            method="spearman", use="pairwise.complete.obs")[,1]),
                    decreasing=TRUE)
cn.asets <- c(cn.asets,
              `names<-`(lapply(c(10, 25, 50, 100),
                               function(p)
                                 names(cn.attr.cor)[1:round(p*length(cn.attr.cor)/100)]),
                        paste("as", c(10, 25, 50, 100), "p.cor", sep="")))

cn.tree.as <-
  lapply(cn.asets,
         function(as)
         {
           tree.d <- rpart(make.formula("ViolentCrimesPerPop", as), cn.train)
           tree.f <- rpart(make.formula("ViolentCrimesPerPop", as), cn.train,
                           minsplit=2, cp=0)
           tree.pmin <- prune(tree.f, cpmin(tree.f$cptable))
           tree.p1sd <- prune(tree.f, cp1sd(tree.f$cptable))
           list(tree.d=tree.d,
                r2.d=r2(predict(tree.d, cn.val), cn.val$ViolentCrimesPerPop),
                tree.pmin=tree.pmin,
                r2.pmin=r2(predict(tree.pmin, cn.val), cn.val$ViolentCrimesPerPop),
                tree.p1sd=tree.p1sd,
                r2.p1sd=r2(predict(tree.p1sd, cn.val), cn.val$ViolentCrimesPerPop))
         })

sapply(cn.tree.as,
       function(ta) c(r2.d=ta$r2.d, r2.pmin=ta$r2.pmin, r2.p1sd=ta$r2.p1sd))

prp(cn.tree.as$as10p$tree.d, varlen=0, faclen=0)

cn.lm.as <-
  lapply(cn.asets,
         function(as)
         {
           lmod <- lm(make.formula("ViolentCrimesPerPop", as), cni.train)
           list(lm=lmod,
                r2=r2(predict(lmod, cni.val), cni.val$ViolentCrimesPerPop))
         })

sapply(cn.lm.as, function(ta) ta$r2)

cn.mtree <- lmrpart(make.formula("ViolentCrimesPerPop", cn.asets[["as10p"]]),
                    cn.train, cp=0.02, skip.attr=TRUE))

r2(predict(cn.mtree, cni.val), cni.val$ViolentCrimesPerPop)
