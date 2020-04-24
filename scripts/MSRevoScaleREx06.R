rxPrivacyControl(FALSE)
options(encoding = "UTF-8"); par(ask=F)
rm(list = ls(all = TRUE))


#library(rattle) # Fancy tree plot

# First check to see if these packages are installed on this machine
pkglist<-c("rpart","rpart.plot","RColorBrewer","party","partykit","caret")
pkgcheck <- pkglist %in% row.names(installed.packages())
pkglist[!pkgcheck]
#COMMMENT the line below if you installed packages earlier e.g on root
for(i in pkglist[!pkgcheck]){install.packages(i,depend=TRUE)}
#this command is for root instalation of missing packages:
if(length(pkglist[!pkgcheck])) cat("install.packages(c(");j=0; for(i in pkglist[!pkgcheck]) { j<-j+1 ;  if(j == length(pkglist[!pkgcheck])) cat(paste('"',i,'"',sep="")) else cat(paste('"',i,'",',sep=""));} ; cat("),depend=TRUE)")
#loading all libraries - necessary to avoid errors of execution
for(i in pkglist) library(i, character.only = TRUE);


if (!suppressPackageStartupMessages(require("MicrosoftML",
                                            quietly = TRUE,
                                            warn.conflicts = FALSE))) {
  stop("The MicrosoftML package does not seem to be installed, so this\n",
       "script cannot be run. \n")
}

Revo.version
Revo.home()
rxGetComputeContext()
rxSetComputeContext()

#RevoScaleR::Rx

# titanic dataset csv file
titanic_csv = "data/titanic.csv"

# dataset column names and types
col_classes = c(
  "PassengerId" = "integer",
  "Survived" = "factor",
  "Pclass" = "factor",
  "Sex" = "factor",
  "Age" = "numeric",
  "SibSp" = "integer",
  "Parch" = "integer",
  "Ticket" = "character",
  "Fare" = "numeric",
  "Cabin" = "character",
  "Embarked" = "factor"
)

# load (in-memory) dataframe
titanic_data = read.csv(titanic_csv, colClasses = col_classes)
head(titanic_data)

# import and reference an external data frame (xdf)
titanic_xdf = "data/titanic.xdf"
rxImport(titanic_csv, titanic_xdf, colClasses = col_classes, overwrite = TRUE)
titanic_xdata <- RxXdfData(titanic_xdf)

# get information about the xdf
rxGetInfo(titanic_xdata, getVarInfo = TRUE, numRows = 2)
rxSummary(~ Age, titanic_xdata)

# data preparation function
prepare_data <- function(data) {
  
  # fix factor levels in Survived
  data$Survived = factor(data$Survived, levels = 0:1, labels = c('No', 'Yes'))
  
  # create a new variable FareToAgeRatio
  data$FareToAgeRatio = data$Fare / data$Age
  
  # handling missing values in Age 
  age_mean = mean(data$Age, na.rm = TRUE)
  data$Age[is.na(data$Age)] <- age_mean
  return(data)
}

# execute a (scalable/parallelizable) data step
rxDataStep(titanic_xdata, titanic_xdata,
           transformFunc = prepare_data,
           overwrite = TRUE)

# show xdf information after data processing
rxGetInfo(titanic_xdata, getVarInfo = TRUE, numRows = 0)
rxSummary( ~ Age, titanic_xdata)

# summarize Survived variable with respect to the Gender variable, then plot it
rxsm = rxSummary( ~ Survived:Sex, titanic_xdata)
genVsSurv <- tidyr::spread(rxsm$categorical[[1]], key = 'Sex', value = 'Counts')
row.names(genVsSurv) <- genVsSurv[, 1]
genVsSurv <- as.matrix(genVsSurv[, -1])
levelplot(prop.table(genVsSurv, 2), xlab = "Survived", ylab = "Gender",
          main = "Survived vs. Gender")

# learn a decision tree using a scalable DT learning algorithm 
rx_decision_tree <- rxDTree(Survived ~ Age + Sex + Fare + Pclass, 
                            data = titanic_xdata, pruneCp = "auto", 
                            reportProgress = 0)

# plot the decision tree
prp(rxAddInheritance(rx_decision_tree),roundint=FALSE)
#library(rattle)
#rattle::fancyRpartPlot(rxAddInheritance(rx_decision_tree))

# perform predictions
test_data = data.frame(Age = c(30,20), Sex = c("male","female"))
predictions = rxPredict(rx_decision_tree, test_data)
head(predictions)

file.remove("data/titanic.xdf")
