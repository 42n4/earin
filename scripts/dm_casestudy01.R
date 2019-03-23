################################################################################
# Casestudy from the Cichosz book "Data mining: explained in R" 
################################################################################
#after pulling new docker 42n4/rstudio run this command
#.rs.restartR()
#http://www.wekaleamstudios.co.uk/supplementary-material/

# Install all packages from Cichosz book "Data mining: explained in R" 
dmrpkglist<-c('dmr.data','dmr.util','dmr.claseval','dmr.stats','dmr.trans','dmr.linreg','dmr.regeval','dmr.dissim',
              'dmr.dectree','dmr.linclas','dmr.disc','dmr.kcenters','dmr.cluseval','dmr.regtree','dmr.attrsel',
              'dmr.ensemble','dmr.kernel','dmr.bayes','dmr.hierclus','dmr.miscost','dmr.rpartutil')
pkgcheck <- dmrpkglist %in% row.names(installed.packages())
dmrpkglist[!pkgcheck]
for(i in dmrpkglist[!pkgcheck]){install_github(paste("42n4/", i, sep=""),force = TRUE)}
dmrpkglist<-c("dmr.util",
              "dmr.trans",
              "dmr.claseval",
              "dmr.rpartutil")
for(i in dmrpkglist) library(i, character.only = TRUE);

pkglist<-c("rpart",
           "rpart.plot",
           "randomForest")
pkgcheck <- pkglist %in% row.names(installed.packages())
pkglist[!pkgcheck]
#COMMMENT the line below if you installed packages earlier e.g on root
for(i in pkglist[!pkgcheck]){install.packages(i,depend=TRUE)}
#this command is for root instalation of missing packages:
if(length(pkglist[!pkgcheck])) cat("install.packages(c(");j=0; for(i in pkglist[!pkgcheck]) { j<-j+1 ;  if(j == length(pkglist[!pkgcheck])) cat(paste('"',i,'"',sep="")) else cat(paste('"',i,'",',sep=""));} ; cat("),depend=TRUE)")
#loading all libraries - necessary to avoid errors of execution
for(i in pkglist) library(i, character.only = TRUE);


#https://kdd.ics.uci.edu/databases/census-income/census-income.html
census <- read.table("data/census-income.data",
                     sep=",", na.strings="?", strip.white=TRUE)
census.test <- read.table("data/census-income.test",
                          sep=",", na.strings="?", strip.white=TRUE)
names(census) <- c("age",
                   "class.of.worker",
                   "detailed.industry.recode",
                   "detailed.occupation.recode",
                   "education",
                   "wage.per.hour",
                   "enroll.in.edu.inst.last.wk",
                   "marital.stat",
                   "major.industry.code",
                   "major.occupation.code",
                   "race",
                   "hispanic.origin",
                   "sex",
                   "member.of.a.labor.union",
                   "reason.for.unemployment",
                   "full.or.part.time.employment.stat",
                   "capital.gains",
                   "capital.losses",
                   "dividends.from.stocks",
                   "tax.filer.stat",
                   "region.of.previous.residence",
                   "state.of.previous.residence",
                   "detailed.household.and.family.stat",
                   "detailed.household.summary.in.household",
                   "instance.weight",
                   "migration.code.change.in.msa",
                   "migration.code.change.in.reg",
                   "migration.code.move.within.reg",
                   "live.in.this.house.1.year.ago",
                   "migration.prev.res.in.sunbelt",
                   "num.persons.worked.for.employer",
                   "family.members.under.18",
                   "country.of.birth.father",
                   "country.of.birth.mother",
                   "country.of.birth.self",
                   "citizenship",
                   "own.business.or.self.employed",
                   "fill.inc.questionnaire.for.veterans.admin",
                   "veterans.benefits",
                   "weeks.worked.in.year",
                   "year",
                   "income")
names(census.test) <- names(census)

#made factors from discrete attributes
ci.discrete <- c("detailed.industry.recode", "detailed.occupation.recode",
                 "own.business.or.self.employed", "veterans.benefits", "year")
for (a in ci.discrete)
{
  census[[a]] <- as.factor(census[[a]])
  census.test[[a]] <- as.factor(census.test[[a]])
}

#get rid of some attributes (sometimes )
census$instance.weight <- NULL
census.test$instance.weight <- NULL

ci.labels <- c("low", "high")
census$income <- factor(ifelse(census$income=="50000+.", "high", "low"),
                        levels=ci.labels)
census.test$income <- factor(ifelse(census.test$income=="50000+.", "high", "low"),
                             levels=ci.labels)

set.seed(12)

rci <- runif(nrow(census))
ci.train <- census[rci>=0.33,]
ci.val <- census[rci<0.33,]
ci.train.small <- census[rci>=0.9,]

#classifier
ci.tree.d <- rpart(income~., ci.train)

#prunning of the tree
#http://scg.sdsu.edu/ctrees_r/
prp(ci.tree.d)
plotcp(ci.tree.d)
printcp(ci.tree.d)
ci.tree.d.pruned<-prune(ci.tree.d, cp = 0.015) #complexity parameter
prp(ci.tree.d.pruned)

#predicted values
ci.tree.d.pred <- predict(ci.tree.d, ci.val, type="c")

#misclassification error for given vectors of predicted and true class labels 
err(ci.tree.d.pred, ci.val$income)

#confusion matrix
#TN FN
#FP TP
ci.tree.d.cm <- confmat(ci.tree.d.pred, ci.val$income)

#true positive 
TP <- ci.tree.d.cm[2,2]
#true negative
TN <- ci.tree.d.cm[1,1]
#false positive (type I error)
FP <- ci.tree.d.cm[2,1]
#false negative (type II error)
FN <- ci.tree.d.cm[1,2]

#another way of calculating misclassification error
(sum(ci.tree.d.cm) - sum(diag(ci.tree.d.cm)))/sum(ci.tree.d.cm)

#misclassification error
(FP+FN)/(TP+TN+FP+FN)

#accuracy
(TP+TN)/(TP+TN+FP+FN)

#true positive rate (recall, sensitivity)
#the ratio of correctly classified as positive to all positive
TP/(TP+FN)

#false positive rate 
#the ratio of incorrectly classified as positive to all negative
FP/(TN+FP)

#precision 
#the ratio of correctly classified as positive to all instances classified as positive 
TP/(TP+FP)

#specificity (1 - fpr) 
#the ratio of correctly classified as negatives to all negative instances 
TN/(TN+FP)

#complementary pairs
#tpr - fpr
#precision - recall
#sensitivity - specificity

ci.tree.d.tpr <- tpr(ci.tree.d.cm)
ci.tree.d.fpr <- fpr(ci.tree.d.cm)

ci.tree.d.fmeasure <- f.measure(ci.tree.d.cm)
#cm.multi<-confmat01(ci.tree.d.pred.multi, ci.val$multiclass)
#rowMeans(sapply(cm.multi,function(cm) c(tpr=tpr(cm), fpr=fpr(cm), fm=f.measure(cm))))
#for weighted confision matrix - wconfmat

#ROC - receiver operating characteristics 
#developed for radar signal detection
#(0,1) - the perfect operating point 
#(1,0) - the worst operating point
#(0,0) - the classifier always predicts class 0 
#(1,1) - the classifier always predicts class 1

ci.tree.d.prob<-predict(ci.tree.d, ci.val)[,2]

pred.s <- ci.tree.d.prob
true.y <- ci.val$income
roc <- function(pred.s, true.y)
{
  cutoff <- Inf  # start with all instances classified as negative
  tp <- fp <- 0
  tn <- sum(2-as.integer(true.y))  # all negative instances
  fn <- sum(as.integer(true.y)-1)  # all positive instances
  rt <- data.frame()
  
  sord <- order(pred.s, decreasing=TRUE)  # score ordering
  #pred.s[sord]
  #as.integer(true.y[sord])
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

ci.tree.d.roc <- roc(ci.tree.d.prob, ci.val$income)
plot(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=8)
auc(ci.tree.d.roc)
#dev.off() #reset graphics if you have different graphical windows

#cutoff=0.5 for predict(..., type="c")
ci.tree.d.cut06 <- ci.tree.d.roc$cutoff[ci.tree.d.roc$tpr>0.6]
#the least cutoff value with tpr > 0.6
ci.tree.d.cut06[1]
#ustep in cutclass
#ustep(ci.tree.d.prob, ci.tree.d.cut06[1])
#default behaviour for predict(..., type="c"): ustep(ci.tree.d.prob, 0.5)

ci.tree.d.cm06 <- confmat(cutclass(ci.tree.d.prob, ci.tree.d.cut06[1], ci.labels),
                          ci.val$income)
ci.tree.d.tpr06 <- tpr(ci.tree.d.cm06)
ci.tree.d.fpr06 <- fpr(ci.tree.d.cm06)
#circle point
points(ci.tree.d.fpr06, ci.tree.d.tpr06, pch=1)

ci.cost2 <- matrix(c(0, 1, 2, 0), nrow=2, byrow=TRUE) 
#cost matrix as a transposed confusion matrix
#0 1
#2 0
#confusion matrix
#TN FN
#FP TP
#cost matrix
#TN   FP=1
#FN=2 TP

#FN=2  cost of mis-classifying a negative example as positive
ci.tree.c2 <- rpart(income~., ci.train, parms=list(loss=ci.cost2))
prp(ci.tree.c2)
ci.tree.c2.pred <- predict(ci.tree.c2, ci.val, type="c")
err(ci.tree.c2.pred, ci.val$income)
# confusion matrix
ci.tree.c2.cm <- confmat(ci.tree.c2.pred, ci.val$income)
# true positives/false negative rates
ci.tree.c2.tpr <- tpr(ci.tree.c2.cm)
ci.tree.c2.fpr <- fpr(ci.tree.c2.cm)

ci.tree.c2.prob <- predict(ci.tree.c2, ci.val)[,2]
ci.tree.c2.roc <- roc(ci.tree.c2.prob, ci.val$income)
plot(ci.tree.c2.roc$fpr, ci.tree.c2.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
points(ci.tree.c2.fpr, ci.tree.c2.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.c2.roc)

ci.tree.c2.cut07 <- ci.tree.c2.roc$cutoff[ci.tree.c2.roc$tpr>0.7]
ci.tree.c2.cm07 <- confmat(cutclass(ci.tree.c2.prob, ci.tree.c2.cut07[1], ci.labels),
                           ci.val$income)
ci.tree.c2.tpr07 <- tpr(ci.tree.c2.cm07)
ci.tree.c2.fpr07 <- fpr(ci.tree.c2.cm07)
points(ci.tree.c2.fpr07, ci.tree.c2.tpr07, pch=1)

ci.tree.w2 <- rpart(income~., ci.train,
                    weights=ifelse(ci.train$income=="high", 2, 1))

ci.cost5 <- matrix(c(0, 1, 5, 0), nrow=2, byrow=TRUE)
ci.tree.c5 <- rpart(income~., ci.train, parms=list(loss=ci.cost5))
# error
ci.tree.c5.pred <- predict(ci.tree.c5, ci.val, type="c")
err(ci.tree.c5.pred, ci.val$income)
# confusion matrix
ci.tree.c5.cm <- confmat(ci.tree.c5.pred, ci.val$income)
# true positive/false positive rates
ci.tree.c5.tpr <- tpr(ci.tree.c5.cm)
ci.tree.c5.fpr <- fpr(ci.tree.c5.cm)
# ROC
ci.tree.c5.prob <- predict(ci.tree.c5, ci.val)[,2]
ci.tree.c5.roc <- roc(ci.tree.c5.prob, ci.val$income)
plot(ci.tree.c5.roc$fpr, ci.tree.c5.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
points(ci.tree.c5.fpr, ci.tree.c5.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.c5.roc)

ci.tree.c5.cut08 <- ci.tree.c5.roc$cutoff[ci.tree.c5.roc$tpr>0.8]
ci.tree.c5.cm08 <- confmat(cutclass(ci.tree.c5.prob, ci.tree.c5.cut08[1], ci.labels),
                           ci.val$income)
ci.tree.c5.tpr08 <- tpr(ci.tree.c5.cm08)
ci.tree.c5.fpr08 <- fpr(ci.tree.c5.cm08)
points(ci.tree.c5.fpr08, ci.tree.c5.tpr08, pch=1)

ci.cost10 <- matrix(c(0, 1, 10, 0), nrow=2, byrow=TRUE)
ci.tree.c10 <- rpart(income~., ci.train, parms=list(loss=ci.cost10))
ci.tree.c10.prob <- predict(ci.tree.c10, ci.val)[,2]
ci.tree.c10.roc <- roc(ci.tree.c10.prob, ci.val$income)
plot(ci.tree.c10.roc$fpr, ci.tree.c10.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate")
# error
ci.tree.c10.pred <- predict(ci.tree.c10, ci.val, type="c")
err(ci.tree.c10.pred, ci.val$income)
# confusion matrix
ci.tree.c10.cm <- confmat(ci.tree.c10.pred, ci.val$income)
# true positive/false positive rates
ci.tree.c10.tpr <- tpr(ci.tree.c10.cm)
ci.tree.c10.fpr <- fpr(ci.tree.c10.cm)
points(ci.tree.c10.fpr, ci.tree.c10.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.c10.roc)

#choosing parameters requiring less computer power
#ci.tree.f <- rpart(income~., ci.train, minsplit=2, cp=0)
ci.tree.f <- rpart(income~., ci.train, minsplit=10, cp=0.0001)
# minimum-error cost-complexity pruning
ci.tree.pmin <- prune(ci.tree.f, cpmin(ci.tree.f$cptable))
prp(ci.tree.pmin)
# 1-sd cost-complexity pruning
ci.tree.p1sd <- prune(ci.tree.f, cp1sd(ci.tree.f$cptable))
prp(ci.tree.p1sd)
c(default=nrow(ci.tree.d$frame), full=nrow(ci.tree.f$frame),
  pruned.min=nrow(ci.tree.pmin$frame), pruned.1sd=nrow(ci.tree.p1sd$frame))

ci.tree.f.pred <- predict(ci.tree.f, ci.val, type="c")
err(ci.tree.f.pred, ci.val$income)

ci.tree.f.cm <- confmat(ci.tree.f.pred, ci.val$income)
ci.tree.f.tpr <- tpr(ci.tree.f.cm)
ci.tree.f.fpr <- fpr(ci.tree.f.cm)

ci.tree.f.prob <- predict(ci.tree.f, ci.val)[,2]
ci.tree.f.roc <- roc(ci.tree.f.prob, ci.val$income)
plot(ci.tree.f.roc$fpr, ci.tree.f.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
points(ci.tree.f.fpr, ci.tree.f.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.f.roc)

ci.tree.pmin.pred <- predict(ci.tree.pmin, ci.val, type="c")
err(ci.tree.pmin.pred, ci.val$income)

ci.tree.pmin.cm <- confmat(ci.tree.pmin.pred, ci.val$income)
ci.tree.pmin.tpr <- tpr(ci.tree.pmin.cm)
ci.tree.pmin.fpr <- fpr(ci.tree.pmin.cm)

ci.tree.pmin.prob <- predict(ci.tree.pmin, ci.val)[,2]
ci.tree.pmin.roc <- roc(ci.tree.pmin.prob, ci.val$income)
plot(ci.tree.pmin.roc$fpr, ci.tree.pmin.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="Minimum error CCP")
points(ci.tree.pmin.fpr, ci.tree.pmin.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.pmin.roc)

ci.tree.p1sd.pred <- predict(ci.tree.p1sd, ci.val, type="c")
err(ci.tree.p1sd.pred, ci.val$income)

ci.tree.p1sd.cm <- confmat(ci.tree.p1sd.pred, ci.val$income)
ci.tree.p1sd.tpr <- tpr(ci.tree.p1sd.cm)
ci.tree.p1sd.fpr <- fpr(ci.tree.p1sd.cm)

ci.tree.p1sd.prob <- predict(ci.tree.p1sd, ci.val)[,2]
ci.tree.p1sd.roc <- roc(ci.tree.p1sd.prob, ci.val$income)
plot(ci.tree.p1sd.roc$fpr, ci.tree.p1sd.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="1-SD CCP")
points(ci.tree.p1sd.fpr, ci.tree.p1sd.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.p1sd.roc)

ci.tree.p1sd.cut08 <- ci.tree.p1sd.roc$cutoff[ci.tree.p1sd.roc$tpr>0.8]
ci.tree.p1sd.cm08 <- confmat(cutclass(ci.tree.p1sd.prob, ci.tree.p1sd.cut08[1],
                                      ci.labels),
                             ci.val$income)
ci.tree.p1sd.tpr08 <- tpr(ci.tree.p1sd.cm08)
ci.tree.p1sd.fpr08 <- fpr(ci.tree.p1sd.cm08)
points(ci.tree.p1sd.fpr08, ci.tree.p1sd.tpr08, pch=1)

#ci.tree.c5f <- rpart(income~., ci.train, minsplit=2, cp=0, parms=list(loss=ci.cost5))
ci.tree.c5f <- rpart(income~., ci.train, minsplit=10, cp=0.0001, parms=list(loss=ci.cost5))
ci.tree.c5f.pred <- predict(ci.tree.c5f, ci.val, type="c")
err(ci.tree.c5f.pred, ci.val$income)

ci.tree.c5f.cm <- confmat(ci.tree.c5f.pred, ci.val$income)
ci.tree.c5f.tpr <- tpr(ci.tree.c5f.cm)
ci.tree.c5f.fpr <- fpr(ci.tree.c5f.cm)

ci.tree.c5f.prob <- predict(ci.tree.c5f, ci.val)[,2]
ci.tree.c5f.roc <- roc(ci.tree.c5f.prob, ci.val$income)
plot(ci.tree.c5f.roc$fpr, ci.tree.c5f.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate")
points(ci.tree.c5f.fpr, ci.tree.c5f.tpr, pch=8)
lines(ci.tree.c5.roc$fpr, ci.tree.c5.roc$tpr, lty=2)
points(ci.tree.c5.fpr, ci.tree.c5.tpr, pch=4)
auc(ci.tree.c5f.roc)

# minimum-error cost-complexity pruning (with cp determined based on ci.tree.f)
ci.tree.c5pmin <- prune(ci.tree.c5f, cpmin(ci.tree.f$cptable))
# 1-sd cost-complexity pruning  (with cp terming based on ci.tree.f)
ci.tree.c5p1sd <- prune(ci.tree.c5f, cp1sd(ci.tree.f$cptable))

ci.tree.c5pmin.pred <- predict(ci.tree.c5pmin, ci.val, type="c")
err(ci.tree.c5pmin.pred, ci.val$income)

ci.tree.c5pmin.cm <- confmat(ci.tree.c5pmin.pred, ci.val$income)
ci.tree.c5pmin.tpr <- tpr(ci.tree.c5pmin.cm)
ci.tree.c5pmin.fpr <- fpr(ci.tree.c5pmin.cm)

ci.tree.c5pmin.prob <- predict(ci.tree.c5pmin, ci.val)[,2]
ci.tree.c5pmin.roc <- roc(ci.tree.c5pmin.prob, ci.val$income)
plot(ci.tree.c5pmin.roc$fpr, ci.tree.c5pmin.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="Minimum error CCP")
points(ci.tree.c5pmin.fpr, ci.tree.c5pmin.tpr, pch=8)
lines(ci.tree.c5.roc$fpr, ci.tree.c5.roc$tpr, lty=2)
points(ci.tree.c5.fpr, ci.tree.c5.tpr, pch=4)
auc(ci.tree.c5pmin.roc)

ci.tree.c5p1sd.pred <- predict(ci.tree.c5p1sd, ci.val, type="c")
err(ci.tree.c5p1sd.pred, ci.val$income)

ci.tree.c5p1sd.cm <- confmat(ci.tree.c5p1sd.pred, ci.val$income)
ci.tree.c5p1sd.tpr <- tpr(ci.tree.c5p1sd.cm)
ci.tree.c5p1sd.fpr <- fpr(ci.tree.c5p1sd.cm)

ci.tree.c5p1sd.prob <- predict(ci.tree.c5p1sd, ci.val)[,2]
ci.tree.c5p1sd.roc <- roc(ci.tree.c5p1sd.prob, ci.val$income)
plot(ci.tree.c5p1sd.roc$fpr, ci.tree.c5p1sd.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="1-SD CCP")
points(ci.tree.c5p1sd.fpr, ci.tree.c5p1sd.tpr, pch=8)
lines(ci.tree.c5.roc$fpr, ci.tree.c5.roc$tpr, lty=2)
points(ci.tree.c5.fpr, ci.tree.c5.tpr, pch=4)
auc(ci.tree.c5p1sd.roc)

ci.tree.c5p1sd.cut085 <- ci.tree.c5p1sd.roc$cutoff[ci.tree.c5p1sd.roc$tpr>0.85]
ci.tree.c5p1sd.cm085 <- confmat(cutclass(ci.tree.c5p1sd.prob,
                                         ci.tree.c5p1sd.cut085[1], ci.labels),
                                ci.val$income)
ci.tree.c5p1sd.tpr085 <- tpr(ci.tree.c5p1sd.cm085)
ci.tree.c5p1sd.fpr085 <- fpr(ci.tree.c5p1sd.cm085)
points(ci.tree.c5p1sd.fpr085, ci.tree.c5p1sd.tpr085, pch=1)

# aggregation (ensure no more than 32 discrete attribute values)
ci.aggm <- agg.all(income~., ci.train.small, 31)
cirf.train <- predict.agg(ci.aggm, ci.train.small)
cirf.val <- predict.agg(ci.aggm, ci.val)
# imputation (ensure no missing values)
cirf.impm <- imp.all(income~., cirf.train)
cirf.train <- predict.imp(cirf.impm, cirf.train)
cirf.val <- predict.imp(cirf.impm, cirf.val)

ci.rf <- randomForest(income~., cirf.train, importance=TRUE)

ci.rf.pred <- predict(ci.rf, cirf.val)
err(ci.rf.pred, cirf.val$income)
ci.rf.cm <- confmat(ci.rf.pred, cirf.val$income)

ci.rf.tpr <- tpr(ci.rf.cm)
ci.rf.fpr <- fpr(ci.rf.cm)

ci.rf.prob <- predict(ci.rf, cirf.val, type="p")[,2]
ci.rf.roc <- roc(ci.rf.prob, cirf.val$income)
plot(ci.rf.roc$fpr, ci.rf.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
points(ci.rf.fpr, ci.rf.tpr, pch=8)
auc(ci.rf.roc)

ci.rf.cut09 <- ci.rf.roc$cutoff[ci.rf.roc$tpr>0.9]
ci.rf.cm09 <- confmat(cutclass(ci.rf.prob, ci.rf.cut09[1], ci.labels), ci.val$income)
ci.rf.tpr09 <- tpr(ci.rf.cm09)
ci.rf.fpr09 <- fpr(ci.rf.cm09)
points(ci.rf.fpr09, ci.rf.tpr09, pch=1)

randomForest::varImpPlot(ci.rf, type=1)

ci.attr.utl <- sort(randomForest::importance(ci.rf)[,1], decreasing=TRUE)
ci.attrs <-
  `names<-`(lapply(c(10, 25, 50, 100),
                   function(p)
                     names(ci.attr.utl)[1:round(p*length(ci.attr.utl)/100)]),
            paste("as", c(10, 25, 50, 100), "p", sep=""))

# models using selected subsets
ci.tree.c5.as <-
  lapply(ci.attrs,
         function(as)
         {
           tree.as <- rpart(make.formula("income", as), ci.train,
                            parms=list(loss=ci.cost5))
           cm.as <- confmat(predict(tree.as, ci.val, type="c"), ci.val$income)
           roc.as <- roc(predict(tree.as, ci.val)[,2], ci.val$income)
           list(tree=tree.as,
                tpr=tpr(cm.as),
                fpr=fpr(cm.as),
                roc=roc.as,
                auc=auc(roc.as))
         })

sapply(ci.tree.c5.as, function(ta) ta$auc)

# default operating point
ci.tree.c5p1sd.test.pred <- predict(ci.tree.c5p1sd, census.test, type="c")
ci.tree.c5p1sd.test.cm <- confmat(ci.tree.c5p1sd.test.pred, census.test$income)
ci.tree.c5p1sd.test.tpr <- tpr(ci.tree.c5p1sd.test.cm)
ci.tree.c5p1sd.test.fpr <- fpr(ci.tree.c5p1sd.test.cm)
# ROC
ci.tree.c5p1sd.test.prob <- predict(ci.tree.c5p1sd, census.test)[,2]
ci.tree.c5p1sd.test.roc <- roc(ci.tree.c5p1sd.test.prob, census.test$income)
plot(ci.tree.c5p1sd.test.roc$fpr, ci.tree.c5p1sd.test.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="Decision tree")
points(ci.tree.c5p1sd.test.fpr, ci.tree.c5p1sd.test.tpr, pch=8)
auc(ci.tree.c5p1sd.roc)
# operating point shifted based on the validation set
ci.tree.c5p1sd.test.cm085 <- confmat(cutclass(ci.tree.c5p1sd.test.prob,
                                              ci.tree.c5p1sd.cut085[1], ci.labels),
                                     census.test$income)
ci.tree.c5p1sd.test.tpr085 <- tpr(ci.tree.c5p1sd.test.cm085)
ci.tree.c5p1sd.test.fpr085 <- fpr(ci.tree.c5p1sd.test.cm085)
points(ci.tree.c5p1sd.test.fpr085, ci.tree.c5p1sd.test.tpr085, pch=1)

# test set preprocessing
cirf.test <- predict.agg(ci.aggm, census.test)
cirf.test <- predict.imp(cirf.impm, cirf.test)
# default operating point
ci.rf.test.pred <- predict(ci.rf, cirf.test)
ci.rf.test.cm <- confmat(ci.rf.test.pred, cirf.test$income)
ci.rf.test.tpr <- tpr(ci.rf.test.cm)
ci.rf.test.fpr <- fpr(ci.rf.test.cm)
# ROC
ci.rf.test.prob <- predict(ci.rf, cirf.test, type="p")[,2]
ci.rf.test.roc <- roc(ci.rf.test.prob, cirf.test$income)
plot(ci.rf.test.roc$fpr, ci.rf.test.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="Random forest")
points(ci.rf.test.fpr, ci.rf.test.tpr, pch=8)
auc(ci.rf.test.roc)
# operating point shifted based on the validation set
ci.rf.test.cm09 <- confmat(cutclass(ci.rf.test.prob, ci.rf.cut09[1], ci.labels),
                           cirf.test$income)
ci.rf.test.tpr09 <- tpr(ci.rf.test.cm09)
ci.rf.test.fpr09 <- fpr(ci.rf.test.cm09)
points(ci.rf.test.fpr09, ci.rf.test.tpr09, pch=1)

prp(prune(ci.tree.c5p1sd, 0.01), varlen=8, faclen=2)

