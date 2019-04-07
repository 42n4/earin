rxPrivacyControl(FALSE)
options(encoding = "UTF-8"); par(ask=F)

if (!suppressPackageStartupMessages(require("MicrosoftML",
                                            quietly = TRUE,
                                            warn.conflicts = FALSE))) {
  stop("The MicrosoftML package does not seem to be installed, so this\n",
       "script cannot be run. \n")
}

# Estimate a logistic regression model
logitModel <- rxLogisticRegression(isCase ~ age + parity + education + spontaneous + induced,
                                   transforms = list(isCase = case == 1),
                                   data = infert)
# Print a summary of the model
summary(logitModel)

# Score to a data frame
scoreDF <- rxPredict(logitModel, data = infert, 
                     extraVarsToWrite = "isCase")

# Compute and plot the Radio Operator Curve and AUC
roc1 <- rxRoc(actualVarName = "isCase", predVarNames = "Probability", data = scoreDF) 
plot(roc1)
rxAuc(roc1)
