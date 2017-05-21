#http://www.wekaleamstudios.co.uk/supplementary-material/
library(dmr.claseval)
library(dmr.util)
library(dmr.trans)

library(rpart)
library(rpart.plot)
library(randomForest)
#https://kdd.ics.uci.edu/databases/census-income/census-income.html
census <- read.table("./census-income.data",
                     sep=",", na.strings="?", strip.white=TRUE)
census.test <- read.table("./census-income.test",
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
prp(ci.tree.d)
plotcp(ci.tree.d)
printcp(ci.tree.d)
ci.tree.d.pruned<-prune(ci.tree.d, cp = 0.015)
prp(ci.tree.d.pruned)
ci.tree.d.roc <- roc(predict(ci.tree.d, ci.val)[,2], ci.val$income)
plot(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=8)
auc(ci.tree.d.roc)

ci.tree.d.cut06 <- ci.tree.d.roc$cutoff[ci.tree.d.roc$tpr>0.6]

#error not found ci.tree.d.prob
ci.tree.d.cm06 <- confmat(cutclass(ci.tree.d.prob, ci.tree.d.cut06[1], ci.labels),
                          ci.val$income)
ci.tree.d.tpr06 <- tpr(ci.tree.d.cm06)
ci.tree.d.fpr06 <- fpr(ci.tree.d.cm06)
points(ci.tree.d.fpr06, ci.tree.d.tpr06, pch=1)

ci.cost2 <- matrix(c(0, 1, 2, 0), nrow=2, byrow=TRUE)
ci.tree.c2 <- rpart(income~., ci.train, parms=list(loss=ct.cost2))

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

ci.tree.w2 <- rpart(income~., census.train,
                    weights=ifelse(census.train$income=="high", 2, 1))

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

ci.tree.c10.prob <- predict(ci.tree.c10, ci.val)[,2]
ci.tree.c10.roc <- roc(ci.tree.c10.prob, ci.val$income)
plot(ci.tree.c10.roc$fpr, ci.tree.c10.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate")
points(ci.tree.c10.fpr, ci.tree.c10.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.c10.roc)

ci.tree.f <- rpart(income~., ci.train, minsplit=2, cp=0)

# minimum-error cost-complexity pruning
ci.tree.pmin <- prune(ci.tree.f, cpmin(ci.tree.f$cptable))
# 1-sd cost-complexity pruning
ci.tree.p1sd <- prune(ci.tree.f, cp1sd(ci.tree.f$cptable))

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

ci.tree.c5f <- rpart(income~., ci.train, minsplit=2, cp=0, parms=list(loss=ci.cost5))

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

varImpPlot(ci.rf, type=1)

ci.attr.utl <- sort(importance(ci.rf)[,1], decreasing=TRUE)
ci.asets <-
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
##############
#20-4.R
##############
library(rpart)
library(rpart.plot)
library(randomForest)
library(cluster)

library(dmr.claseval)
library(dmr.cluseval)
library(dmr.trans)
library(dmr.util)

covtype <- read.table("../Data/covtype.data", sep=",",
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
