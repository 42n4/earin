################################################################################
# Casestudy from the Cichosz book "Data mining: explained in R" 
################################################################################
#after pulling new docker 42n4/rstudio run this command
#.rs.restartR()
##############
#20-4.R
##############

# Install all packages from Cichosz book "Data mining: explained in R" 
dmrpkglist<-c('dmr.data','dmr.util','dmr.claseval','dmr.stats','dmr.trans','dmr.linreg','dmr.regeval','dmr.dissim',
              'dmr.dectree','dmr.linclas','dmr.disc','dmr.kcenters','dmr.cluseval','dmr.regtree','dmr.attrsel',
              'dmr.ensemble','dmr.kernel','dmr.bayes','dmr.hierclus','dmr.miscost','dmr.rpartutil')
pkgcheck <- dmrpkglist %in% row.names(installed.packages())
dmrpkglist[!pkgcheck]
for(i in dmrpkglist[!pkgcheck]){install_github(paste("42n4/", i, sep=""),force = TRUE)}
dmrpkglist<-c("dmr.util",
              "dmr.trans",
              "dmr.cluseval",
              "dmr.claseval",
              "dmr.rpartutil")
for(i in dmrpkglist) library(i, character.only = TRUE);

# First check to see if these packages are installed on this machine
pkglist<-c("rpart",
           "rpart.plot",
           "randomForest",
           "cluster")
pkgcheck <- pkglist %in% row.names(installed.packages())
pkglist[!pkgcheck]
#COMMMENT the line below if you installed packages earlier e.g on root
for(i in pkglist[!pkgcheck]){install.packages(i,depend=TRUE)}
#this command is for root instalation of missing packages:
if(length(pkglist[!pkgcheck])) cat("install.packages(c(");j=0; for(i in pkglist[!pkgcheck]) { j<-j+1 ;  if(j == length(pkglist[!pkgcheck])) cat(paste('"',i,'"',sep="")) else cat(paste('"',i,'",',sep=""));} ; cat("),depend=TRUE)")
#loading all libraries - necessary to avoid errors of execution
for(i in pkglist) library(i, character.only = TRUE);

#https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/

covtype <- read.table("data/covtype.data", sep=",",
                      col.names=c("Elevation",
                                  "Aspect",
                                  "Slope",
                                  "Horizontal.Distance.To.Hydrology",
                                  "Vertical.Distance.To.Hydrology",
                                  "Horizontal.Distance.To.Roadways",
                                  "Hillshade.9am",
                                  "Hillshade.Noon",
                                  "Hillshade.3pm",
                                  "Horizontal.Distance.To.Fire.Points",
                                  paste("Wilderness.Area", 1:4, sep=""),
                                  paste("Soil.Type", 1:40, sep=""),
                                  "Cover.Type"))

ct.input.attrs <- setdiff(names(covtype), "Cover.Type")

covtype$Cover.Type <- as.factor(covtype$Cover.Type)

set.seed(12)

rct <- runif(nrow(covtype))
ct.train <- covtype[rct>=0.5,]
ct.val <- covtype[rct>=0.25 & rct<0.5,]
ct.test <- covtype[rct<0.25,]

table(ct.train$Cover.Type)/nrow(ct.train)

ct.tree.d <- rpart(Cover.Type~., ct.train, xval=0)
ct.tree.d.pred <- predict(ct.tree.d, ct.val, type="c")
err(ct.tree.d.pred, ct.val$Cover.Type)
confmat(ct.tree.d.pred, ct.val$Cover.Type)

ct.tree.d.cm01 <- confmat01(ct.tree.d.pred, ct.val$Cover.Type)
ct.tree.d.tpfp <- sapply(ct.tree.d.cm01,
                         function(cm) c(tpr=tpr(cm), fpr=fpr(cm), fm=f.measure(cm)))

rowMeans(ct.tree.d.tpfp)

apply(ct.tree.d.tpfp, 1, weighted.mean, table(ct.val$Cover.Type))

ct.tree.f <- rpart(Cover.Type~., ct.train, minsplit=2, cp=0)

ct.tree.f.pred <- predict(ct.tree.f, ct.val, type="c")
ct.tree.f.cm01 <- confmat01(ct.tree.f.pred, ct.val$Cover.Type)
ct.tree.f.tpfp <- sapply(ct.tree.f.cm01,
                         function(cm) c(tpr=tpr(cm), fpr=fpr(cm), fm=f.measure(cm)))
rowMeans(ct.tree.f.tpfp)

ct.tree.pmin <- prune(ct.tree.f, cpmin(ct.tree.f$cptable))
ct.tree.pmin.pred <- predict(ct.tree.pmin, ct.val, type="c")
ct.tree.pmin.cm01 <- confmat01(ct.tree.pmin.pred, ct.val$Cover.Type)
ct.tree.pmin.tpfp <- sapply(ct.tree.pmin.cm01,
                            function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                           fm=f.measure(cm)))
rowMeans(ct.tree.pmin.tpfp)

ct.tree.p1sd <- prune(ct.tree.f, cp1sd(ct.tree.f$cptable))
ct.tree.p1sd.pred <- predict(ct.tree.p1sd, ct.val, type="c")
ct.tree.p1sd.cm01 <- confmat01(ct.tree.p1sd.pred, ct.val$Cover.Type)
ct.tree.p1sd.tpfp <- sapply(ct.tree.p1sd.cm01,
                            function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                           fm=f.measure(cm)))
rowMeans(ct.tree.p1sd.tpfp)

c(default=nrow(ct.tree.d$frame), full=nrow(ct.tree.f$frame),
  pruned.min=nrow(ct.tree.pmin$frame), pruned.1sd=nrow(ct.tree.p1sd$frame))

ct.tree.w <- rpart(Cover.Type~., ct.train, xval=0,
                   parms=list(prior=rep(1/nlevels(ct.train$Cover.Type),
                                        nlevels(ct.train$Cover.Type))))

ct.tree.w.pred <- predict(ct.tree.w, ct.val, type="c")
ct.tree.w.cm01 <- confmat01(ct.tree.w.pred, ct.val$Cover.Type)
ct.tree.w.tpfp <- sapply(ct.tree.w.cm01,
                         function(cm) c(tpr=tpr(cm), fpr=fpr(cm), fm=f.measure(cm)))
rowMeans(ct.tree.w.tpfp)

ct.tree.w.f <- rpart(Cover.Type~., ct.train, minsplit=2, cp=0,
                     parms=list(prior=rep(1/nlevels(ct.train$Cover.Type),
                                          nlevels(ct.train$Cover.Type))))

ct.tree.w.f.pred <- predict(ct.tree.w.f, ct.val, type="c")
ct.tree.w.f.cm01 <- confmat01(ct.tree.w.f.pred, ct.val$Cover.Type)
ct.tree.w.f.tpfp <- sapply(ct.tree.w.f.cm01,
                           function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                          fm=f.measure(cm)))
rowMeans(ct.tree.w.f.tpfp)

ct.tree.w.p1sd <- prune(ct.tree.w.f, cp1sd(ct.tree.w.f$cptable))
ct.tree.w.pmin <- prune(ct.tree.w.f, cpmin(ct.tree.w.f$cptable))

ct.tree.w.pmin <- rpart(Cover.Type~., ct.train, xval=0,
                        minsplit=2, cp=cpmin(ct.tree.w.f$cptable),
                        parms=list(prior=rep(1/nlevels(ct.train$Cover.Type),
                                             nlevels(ct.train$Cover.Type))))
ct.tree.w.p1sd <- rpart(Cover.Type~., ct.train, xval=0,
                        minsplit=2, cp=cp1sd(ct.tree.w.f$cptable),
                        parms=list(prior=rep(1/nlevels(ct.train$Cover.Type),
                                             nlevels(ct.train$Cover.Type))))

ct.tree.w.pmin.pred <- predict(ct.tree.w.pmin, ct.val, type="c")
ct.tree.w.pmin.cm01 <- confmat01(ct.tree.w.pmin.pred, ct.val$Cover.Type)
ct.tree.w.pmin.tpfp <- sapply(ct.tree.w.pmin.cm01,
                              function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                             fm=f.measure(cm)))
rowMeans(ct.tree.w.pmin.tpfp)

ct.tree.w.p1sd.pred <- predict(ct.tree.w.p1sd, ct.val, type="c")
ct.tree.w.p1sd.cm01 <- confmat01(ct.tree.w.p1sd.pred, ct.val$Cover.Type)
ct.tree.w.p1sd.tpfp <- sapply(ct.tree.w.p1sd.cm01,
                              function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                             fm=f.measure(cm)))
rowMeans(ct.tree.w.p1sd.tpfp)

c(weighted.pruned.min=nrow(ct.tree.w.pmin$frame),
  weighted.pruned.1sd=nrow(ct.tree.w.p1sd$frame))

rp.1k <- multi.class(rpart, predf=function(...) predict(...)[,2],
                     encode=multi.enc.1ofk, decode=multi.dec.1ofk)

ct.tree.1k <- rp.1k$alg(Cover.Type~., ct.train, xval=0)

ct.tree.1k.pred <- rp.1k$predict(ct.tree.1k, ct.val)
ct.tree.1k.cm01 <- confmat01(ct.tree.1k.pred, ct.val$Cover.Type)
ct.tree.1k.tpfp <- sapply(ct.tree.1k.cm01,
                          function(cm) c(tpr=tpr(cm), fpr=fpr(cm), fm=f.measure(cm)))
rowMeans(ct.tree.1k.tpfp)

sapply(ct.tree.1k$binmodels, function(m) nrow(m$frame))

ct.tree.1k.cp <- rp.1k$alg(Cover.Type~., ct.train, xval=0, minsplit=2, cp=1e-5)

ct.tree.1k.cp.pred <- rp.1k$predict(ct.tree.1k.cp, ct.val)
ct.tree.1k.cp.cm01 <- confmat01(ct.tree.1k.cp.pred, ct.val$Cover.Type)
ct.tree.1k.cp.tpfp <- sapply(ct.tree.1k.cp.cm01,
                             function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                            fm=f.measure(cm)))
rowMeans(ct.tree.1k.cp.tpfp)

rp.1k.pmin <- multi.class(rpart.pmin, predf=function(...) predict(...)[,2],
                          encode=multi.enc.1ofk, decode=multi.dec.1ofk)
ct.tree.1k.pmin <- rp.1k.pmin$alg(Cover.Type~., ct.train)

ct.tree.1k.pmin.pred <- rp.1k$predict(ct.tree.1k.pmin, ct.val)
ct.tree.1k.pmin.cm01 <- confmat01(ct.tree.1k.pmin.pred, ct.val$Cover.Type)
ct.tree.1k.pmin.tpfp <- sapply(ct.tree.1k.pmin.cm01,
                               function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                              fm=f.measure(cm)))
rowMeans(ct.tree.1k.pmin.tpfp)

ct.tree.pmin.test.pred <- predict(ct.tree.pmin, ct.test, type="c")
ct.tree.pmin.test.cm01 <- confmat01(ct.tree.pmin.test.pred, ct.test$Cover.Type)
ct.tree.pmin.test.tpfp <- sapply(ct.tree.pmin.test.cm01,
                                 function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                                fm=f.measure(cm)))
rowMeans(ct.tree.pmin.test.tpfp)

ct.tree.w.pmin.test.pred <- predict(ct.tree.w.pmin, ct.test, type="c")
ct.tree.w.pmin.test.cm01 <- confmat01(ct.tree.w.pmin.test.pred, ct.test$Cover.Type)
ct.tree.w.pmin.test.tpfp <- sapply(ct.tree.w.pmin.test.cm01,
                                   function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                                  fm=f.measure(cm)))
rowMeans(ct.tree.w.pmin.test.tpfp)

ct.nrmm <- nrm.all(Cover.Type~., ct.train)
ctn.train <- predict.nrm(ct.nrmm, ct.train)
ctn.val <- predict.nrm(ct.nrmm, ct.val)
ctn.test <- predict.nrm(ct.nrmm, ct.test)

ctn.cla <-
  `names<-`(lapply(2:10, function(k)
    clara(ctn.train[,ct.input.attrs], k,
          samples=100, sampsize=200, keep.data=FALSE)), 2:10)

plot(2:10, sapply(ctn.cla, function(cm) cm$silinfo$avg.width),
     type="l", xlab="k", ylim=c(0, 0.5))
lines(2:10, sapply(ctn.cla, function(cm) sd(cm$silinfo$clus.avg.widths)), lty=2)
legend("bottomright", legend=c("average silhouette width",
                               "sd(cluster silhouette widths)"), lty=1:2)

par(mfrow=c(1, 3))
plot(silhouette(ctn.cla[["2"]]), main="k=2")
plot(silhouette(ctn.cla[["7"]]), main="k=7")
plot(silhouette(ctn.cla[["10"]]), main="k=10")

ctn.cla7.pred <- predict(ctn.cla[["7"]], ctn.val[,ct.input.attrs])

ct.cla7.tree <- rpart(make.formula("as.factor(ctn.cla[[\"7\"]]$clustering)",
                                   ct.input.attrs),
                      ct.train, xval=0)

ct.cla7.tree.pred <- predict(ct.cla7.tree, ct.val, type="c")
ct.cla7.tree.cm01 <- confmat01(ct.cla7.tree.pred, as.factor(ctn.cla7.pred))
ct.cla7.tree.tpfp <- sapply(ct.cla7.tree.cm01,
                            function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                           fm=f.measure(cm)))
rowMeans(ct.cla7.tree.tpfp)

prp(ct.cla7.tree, varlen=0, faclen=0)
