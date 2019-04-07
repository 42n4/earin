rxPrivacyControl(FALSE)
options(encoding = "UTF-8"); par(ask=F)
# We'll use some well known CRAN libraries for this example
# mlbench: Use the builtin standard UCI breast cancer dataset
# caret: Use builtin function to create train and test datasets from a single dataset
# dplyr: Select specific columns from the dataframe

# First check to see if these packages are installed on this machine
pkglist<-c("mlbench","caret","dplyr")
pkgcheck <- pkglist %in% row.names(installed.packages())
pkglist[!pkgcheck]
#COMMMENT the line below if you installed packages earlier e.g on root
for(i in pkglist[!pkgcheck]){install.packages(i,depend=TRUE)}
#this command is for root instalation of missing packages:
if(length(pkglist[!pkgcheck])) cat("install.packages(c(");j=0; for(i in pkglist[!pkgcheck]) { j<-j+1 ;  if(j == length(pkglist[!pkgcheck])) cat(paste('"',i,'"',sep="")) else cat(paste('"',i,'",',sep=""));} ; cat("),depend=TRUE)")
#loading all libraries - necessary to avoid errors of execution
for(i in pkglist) library(i, character.only = TRUE);

# Load the MicrosoftML library
library(MicrosoftML)
# Load the breast cancer dataset in the mlbench library
data(BreastCancer)

# Examine the data
head(BreastCancer)
summary(BreastCancer)
# Since the label column (Class) is text, and the ML algorithm can train only
# on numeric labels, create a new Label column such that:
# benign <- 0, malignant <- 1
BreastCancer$Label[BreastCancer$Class == "benign"] <- 0
BreastCancer$Label[BreastCancer$Class == "malignant"] <- 1

# We don't need the Id and Class columns, so let's drop them
breastCancerDS <- dplyr::select(BreastCancer, -Id, -Class)

#5. Partition the dataset into train and test datasets using caret

# Partition the dataset 75%/25% split in order to create a train and test dataset
bCDS <- createDataPartition(y = breastCancerDS$Label, p = .75,  list = FALSE)

# Get the train and test dataset
trainDS <- breastCancerDS[bCDS, ] # This gives us the 75%
testDS <- breastCancerDS[-bCDS, ] # This gives us the remaining 25%

#6. Train a model using rxFastLinear

#Using the training dataset obtained from the previous step, we'll use the rxFastLinear MicrosoftML algorithm to train a model. A unique feature of the rxFastLinear is that the L1 and L2 regularization factors are automatically determined from the dataset.

# Now train using rxFastLinear
# We'll use default parameters
# NOTE: One of the cool things about rxFastLinear is that L1 and L2 are automagically determined
# NOTE: from the training dataset
allVars <- names(trainDS)

model <- rxFastLinear(formula(paste("Label ~", paste(allVars, collapse = " + "))), 
                      trainDS)

#7. Predict using the test dataset

# Let's evaluate the model using the test dataset
score <- rxPredict(model, data = testDS, extraVarsToWrite = "Label")

head(score)

#8. Evaluate the performance of the model

# Let's look at the AUC and ROC
rxRocCurve(actualVarName = "Label", predVarNames = "Probability.1", data = score)

