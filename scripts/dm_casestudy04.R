#after pulling new docker 42n4/rstudio run this command
#.rs.restartR()
start.time <- Sys.time()
#For students with some doubts about the suggested caret solution, 
#I prepared the exemplary case study only with a use of caret - a wrapper of many classifiers and not only:
#https://github.com/pwasiewi/earin/blob/master/scripts/dm_casestudy04.R
#https://github.com/pwasiewi/dokerz/blob/master/rstudio/dm_casestudy04.R
#Try to modify some parameters e.g. in a train function:tuneLength,metric,preProc and so on.
#For your data try also other classifiers from site:
#http://topepo.github.io/caret/train-models-by-tag.html
#Their parameters mentioned in their descriptions are automatically estimated 
#(in default grid tunes) e.g. with the repeated kfold crossvalidation in the exemplary script.
#rpart works with weights or priors for unbalanced classes
#http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
#try different values of weights or priors and minsplit and cp, 
#try to shift working point with cutclass to get better results
#try other classifiers with priors or weights for class classes
#at the end compare all roc curves of all used classifiers
#e.g. https://rpubs.com/chengjiun/52658
#some other measures are mentioned here:
#https://geekoverdose.wordpress.com/2014/07/25/svm-classification-example-with-performance-measures-using-r-caret/
#
#http://www.wekaleamstudios.co.uk/supplementary-material/
#http://www.r-tutor.com/taxonomy/term/286 #GPU SVM

dmrpkglist<-c('dmr.data','dmr.util','dmr.claseval','dmr.stats','dmr.trans','dmr.linreg','dmr.regeval','dmr.dissim',
              'dmr.dectree','dmr.linclas','dmr.disc','dmr.kcenters','dmr.cluseval','dmr.regtree','dmr.attrsel',
              'dmr.ensemble','dmr.kernel','dmr.bayes','dmr.hierclus','dmr.miscost','dmr.rpartutil')
#install_github(paste("42n4/", dmrpkglist, sep=""),force = TRUE)
library(dmr.claseval) #dmr from the Cichosz book "Data mining: explained in R" 
library(dmr.util)
library(dmr.trans)    
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(corrplot)			# plot correlations
library(doParallel)		# parallel processing
library(dplyr)        # Used by caret
library(gbm)				  # GBM Models
library(pROC)				  # plot the ROC curve
library(xgboost) 
library(doParallel)
library(gbm)
library(party)
library(partykit)
library(doParallel)

##########################################################################################################
#My functions
##########################################################################################################


#function removes columns with zero data
remove_colzero<-function(df_test){
  df_test[,colSums (df_test^2)!=0]
}

#this function removes na from numeric and factor columns
remove_na<-function(df_test){
  for (var in 1:ncol(df_test)) {
    if (class(df_test[,var])=="numeric") {
      df_test[is.na(df_test[,var]),var] <- mean(df_test[,var], na.rm = TRUE)
    } else if (class(df_test[,var]) %in% c("character", "factor")) {
      df_test[is.na(df_test[,var]),var] <- Mode(df_test[,var], na.rm = TRUE)
    }
  }
  df_test
}


#function searches through a correlation matrix and 
#uses a vector of integers corresponding to columns to remove to reduce pair-wise correlations.
remove_correlated<-function(df_test,mycutoff=0.8){
  cordf2 = cor(df_test)#findCorrelation is a function built into the caret package 
  hc = findCorrelation(cordf2, cutoff=mycutoff, verbose = FALSE) # put any value as a "cutoff" ,A numeric value for the pair-wise absolute correlation cutoff
  hc = sort(hc)#after find Correlation data we sort it
  df_test[,-c(hc)]
}

get_cm<-function(classifier,dataset.val, mytype="c"){
  ci.train.rpart <- predict(classifier, dataset.val, type = mytype)
  ci.train.rpart.cm <- confmat(ci.train.rpart, dataset.val[,ncol(dataset.val)])
  ci.train.rpart.cm
}


show_roc<-function(classifier,dataset.val,mainstr="ROC"){
  ci.train.rpart.prob <- predict(classifier, dataset.val, type="prob")[,2]
  ci.train.rpart.roc <- dmr.claseval::roc(ci.train.rpart.prob, dataset.val[,ncol(dataset.val)])
  plot(ci.train.rpart.roc$fpr, ci.train.rpart.roc$tpr, type="l", xlab="FP rate", ylab="TP rate", main=mainstr)
  ci.train.rpart.roc
}

get_time<-function(){
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  start.time <<- Sys.time()
  time.taken
}

#significant columns
columns_boosted<-function(df_test){
  
  #http://www.rmdk.ca/boosting_forests_bagging.html
  boost <<- gbm(as.formula(paste(names(df_test)[ncol(df_test)],"~.",sep="")), data=df_test, 
                distribution = 'gaussian', #for multi class classs
                #distribution='bernoulli', #for class 0 and 1, but worse than gaussian
                n.trees = 100, 
                interaction.depth = 4) #names(df_test)[ncol(df_test)]
  names_chosen<-c(as.character(summary(boost)[1:8,1]),names(df_test)[ncol(df_test)])
  names_chosen
}

#Mode the frequent attribute values
Mode <- function (x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- ">1 mode"
  return(xmode)
}

##########################################################################################################
##########################################################################################################
# Set up to do parallel processing   
registerDoParallel(4)		# Register a parallel backend for train
getDoParWorkers()

#data description
#http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.names
#download.file('http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data', 'magic.data')
magic = read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data", header = FALSE,
                   col.names = c('fLength', 'fWidth', 'fSize', 'fConc', 'fConc1','fAsym'
                                 , "fM3Long", "fM3Trans", "fAlpha", "fDist", "class") )


summary(magic)
names(magic)  #check every column name
magic$class <- factor(magic$class)
table(magic$class)

##########################################################################################################
#remove columns with zeros
#magic1<-remove_colzero(magic)
#names(magic1)

##########################################################################################################
magic1<-remove_na(magic)
names(magic1)
sum(magic1$class=="h")/nrow(magic1)*100

get_time()

##########################################################################################################
#remove any variables that are correlated to each other more than 0.8
#Remove highly correlated variables  
magic2<-remove_correlated(magic1[,-ncol(magic1)], 0.9)
magic1<-cbind(magic2,magic1[ncol(magic1)])

##########################################################################################################
#remove not boosted attributes (columns)
names_chosen <- columns_boosted(magic1)
magic2<-magic1[,names_chosen]
names(magic2) 
sum(magic2$class=="h")/nrow(magic2)*100

get_time()

##########################################################################################################
#the same as loop, 2 means columns, 1 means rows, z means the given row or column
#unique values
apply(magic2,2,function(z){length(unique(z))})  

##########################################################################################################
#upsample rows from unbalanced classes to make them balanced
#https://topepo.github.io/caret/subsampling-for-class-imbalances.html
magic3 <- upSample(x = magic2[, -ncol(magic2)], y = magic2$class)
nrow(magic3)
names(magic3)
magic3$class<-magic3$Class
magic3$Class<-NULL
table(magic2$class)
table(magic3$class)

#return of unbalanced data
magic3<-magic2

get_time()

##########################################################################################################
set.seed(1234)#which is useful for creating simulations or random objects that can be reproduced.
rci <- runif(nrow(magic3))
ci.train <- magic3[rci>=0.33,]
ci.val <- magic3[rci<0.33,]

table(ci.train$class)
names(ci.train)
length(ci.train)
ncol(ci.train)
nrow(ci.train)
sum(ci.train$class=="h")/nrow(ci.train)*100
levels(ci.train$class)
weights <- as.numeric(ci.train$class)
for(val in unique(weights)) {weights[weights==val]=1/sum(weights==val)*length(weights)/2} # normalized to sum to length(samples)
weights100 <- round(weights*100,0)
table(weights100)

##########################################################################################################

ProbC=as.numeric(table(ci.train$class)[2]/nrow(ci.train))
ProbC=1-ifelse(ProbC<0.2, 0.2, ProbC)

get_time()

##########################################################################################################
#3 plots in two column
dev.off(); par(mfrow = c(3, 2))
#rpart works with weigths or priors  - prior probabilities of output class classes: sum(priors)=1
#maximum size of tree with minsplit=2, cp=0 and priors

##########################################################################################################
#read about caret package, use boosting, bagging, k-fold crossvalidating
ctrl <- trainControl(method = "repeatedcv"
                     , number = 3
                     , repeats = 3
                     , classProbs = TRUE
                     , allowParallel=TRUE
                     #, summaryFunction=twoClassSummary
)
caretrpart <- NULL
caretrpart <- train(method = 'rpart2', 
                    x = ci.train[,-ncol(ci.train)], 
                    y = ci.train[,ncol(ci.train)],           
                    parms = list(prior = c(ProbC, 1 - ProbC))
                    #, tuneGrid = data.frame(cp = c(0.0005669276, 0.0006669276, 0.0007669276))
                    #, tuneGrid = data.frame(cp = seq(0.0004,0.0005,0.00001))
                    , control = rpart.control(minsplit = 10)
                    #,type = "Classification"
                    #,tuneLength=10
                    #,metric="ROC"
                    #,preProc = c("center", "scale")
                    #, trControl= trainControl(method="boot", number=10)
                    ,trControl = ctrl
)
#plot(caretrpart)
#summary(caretrpart)
#prp(caretrpart$finalModel)
ci.train.caretrpart.cm <- get_cm(caretrpart,ci.val,"raw")
ci.train.caretrpart.cm
ci.train.caretrpart.roc <- show_roc(caretrpart,ci.val, "ROC caret rpart2")
dmr.claseval::auc(ci.train.caretrpart.roc)

get_time()

##########################################################################################################
#ctree only tunes over mincriterion and ctree2 tunes over maxdepth (while fixing mincriterion = 0)
caretctree <- train(method = 'ctree2', 
                    x = ci.train[,-ncol(ci.train)], 
                    y = ci.train[,ncol(ci.train)]   
                    , weights=weights100
                    #, trControl= trainControl(method = "repeatedcv", number = 10, repeats = 10)
                    #,controls=ctree_control(minbucket=3)
                    #,controls=ctree_control(minbucket=3, maxdepth = 3, maxsurrogate = 3, mincriterion=0.95,savesplitstats=FALSE)
                    ,trControl = ctrl
)
#https://stats.stackexchange.com/questions/171301/interpreting-ctree-partykit-output-in-r

#plot(caretctree,type="simple")
ci.train.caretctree.cm <- get_cm(caretctree,ci.val,"raw")
ci.train.caretctree.cm
ci.train.caretctree.roc <- show_roc(caretctree,ci.val, "ROC caret ctree")
dmr.claseval::auc(ci.train.caretctree.roc)

get_time()

##########################################################################################################
caretlda <- NULL
caretlda <- train(method = 'lda', 
                  x = ci.train[,-ncol(ci.train)], 
                  y = ci.train[,ncol(ci.train)]
                  , parms = list(prior = c(ProbC, 1 - ProbC))
                  #, weights=weights100
                  ,trControl = ctrl
)
ci.train.caretlda.cm <- get_cm(caretlda,ci.val,"raw")
ci.train.caretlda.cm
ci.train.caretlda.roc <- show_roc(caretlda,ci.val, "ROC caret lda")
dmr.claseval::auc(ci.train.caretlda.roc)

get_time()

##########################################################################################################
grid <- expand.grid(interaction.depth=c(1,2,3),     # Depth of variable interactions
                    n.trees=c(140,150,160),	        # Num trees to fit
                    shrinkage=c(0.01,0.1),		      # Try 2 values for learning rate 
                    n.minobsinnode = c(10,20))
caretgbm <- NULL
caretgbm <- train(method = 'gbm', 
                  x = ci.train[,-ncol(ci.train)], 
                  y = ci.train[,ncol(ci.train)]
                  #,parms = list(prior = c(ProbC, 1 - ProbC))
                  #,tuneGrid=grid #usually do it automatically with a default grid
                  ,weights=weights100
                  ,trControl = ctrl
                  ,verbose=FALSE
)
caretgbm$bestTune
#plot(caretgbm)
ci.train.caretgbm.cm <- get_cm(caretgbm,ci.val,"raw")
ci.train.caretgbm.cm
ci.train.caretgbm.roc <- show_roc(caretgbm,ci.val, "ROC caret gbm")
ci.train.caretgbm.cm

get_time()


##########################################################################################################
caretreebag <- NULL
caretreebag <- train(method = 'treebag', 
                     x = ci.train[,-ncol(ci.train)], 
                     y = ci.train[,ncol(ci.train)]
                     #,parms = list(prior = c(ProbC, 1 - ProbC))
                     #,tuneGrid=grid
                     #, metric="ROC"
                     ,weights=weights100
                     ,trControl = ctrl
                     #,verbose=FALSE
)
ci.train.caretreebag.cm <- get_cm(caretreebag,ci.val,"raw")
ci.train.caretreebag.cm
ci.train.caretreebag.roc <- show_roc(caretreebag,ci.val, "ROC caret treebag")
dmr.claseval::auc(ci.train.caretreebag.roc)

get_time()

##########################################################################################################

ci.train.caretrpart.cm
ci.train.caretctree.cm
ci.train.caretlda.cm
ci.train.caretgbm.cm
ci.train.caretreebag.cm

dmr.claseval::auc(ci.train.caretrpart.roc)
dmr.claseval::auc(ci.train.caretctree.roc)
dmr.claseval::auc(ci.train.caretlda.roc)
dmr.claseval::auc(ci.train.caretgbm.roc)
dmr.claseval::auc(ci.train.caretreebag.roc)

get_time()

results <- resamples(list(CRP=caretrpart, CCT=caretctree, CLD=caretlda, CGB=caretgbm, CTB=caretreebag))
# summarize the distributions
summary(results)
Sys.sleep(5)                                # 5 second pause
# boxplots of results
bwplot(results)
Sys.sleep(5)                                # 5 second pause
parallelplot(results)
Sys.sleep(5)                                # 5 second pause

##########################################################################################################
#again for unbalanced data
#3 plots in two column
dev.off(); par(mfrow = c(3, 2))
rci1 <- runif(nrow(magic2))
ci.train1 <- magic2[rci1>=0.33,]
ci.val1 <- magic2[rci1<0.33,]
ci.train.caretrpart.roc <- show_roc(caretrpart,ci.val1, "ROC caret rpart2")
ci.train.caretctree.roc <- show_roc(caretctree,ci.val1, "ROC caret ctree")
ci.train.caretlda.roc <- show_roc(caretlda,ci.val1, "ROC caret lda")
ci.train.caretgbm.roc <- show_roc(caretgbm,ci.val1, "ROC caret gbm")
ci.train.caretreebag.roc <- show_roc(caretreebag,ci.val1, "ROC caret treebag")



##########################################################################################################
#BELOW tests for caret classifier models: for several thousand of random chosen rows from bigger data

#failed (maybe you will succeed): rda, rpartScore, rpartCost, ctree, xgbTree, blackboost
#deepboost
#time consuming: cforest, adaboost
#succeded: rpart2, ctree2, gbm, treebag, LogitBoost, rotationForest, rotationForestCp
#glmboost
#642 rows TH=0.05: adaboost, AdaBoost.M1, AdaBag, ada, ORFsvm, evtree, nodeHarvest,
#
##########################################################################################################


# Set up to do parallel processing   
#registerDoParallel(4)		# Register a parallel backend for train
registerDoParallel(3,cores=3)
getDoParWorkers()
ctrl <- trainControl(method = "repeatedcv"
                     , number = 3
                     , repeats = 3
                     , classProbs = TRUE
                     , allowParallel=TRUE
                     #, summaryFunction=twoClassSummary
)
nrow(ci.train)
get_time()
caretmodel <- NULL
set.seed(12345)
rci2 <- runif(nrow(ci.train))
TH<-0.005
table(ci.train[rci2<TH,ncol(ci.train)])
caretmodel <- train(method = 'glmboost', 
                    x = ci.train[rci2<TH,-ncol(ci.train)], 
                    y = ci.train[rci2<TH,ncol(ci.train)]
                    #,parms = list(prior = c(ProbC, 1 - ProbC))
                    ,weights=weights100[rci2<TH]
                    #,preProcess = c('center', 'scale')
                    #, metric="ROC"
                    ,trControl = ctrl
)
ci.train.caretmodel.cm <- get_cm(caretmodel,ci.val1,"raw")
ci.train.caretmodel.cm
ci.train.caretmodel.roc <- show_roc(caretmodel,ci.val1, "ROC caret test model")
dmr.claseval::auc(ci.train.caretmodel.roc)
get_time()

#getModelInfo(model = "treebag", regex = FALSE)[[1]]

