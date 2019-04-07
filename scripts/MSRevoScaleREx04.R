#######################################################################################
# Multi-class logistic regression  
rxPrivacyControl(FALSE)
options(encoding = "UTF-8"); par(ask=F)

if (!suppressPackageStartupMessages(require("MicrosoftML",
                                            quietly = TRUE,
                                            warn.conflicts = FALSE))) {
  stop("The MicrosoftML package does not seem to be installed, so this\n",
       "script cannot be run. \n")
}

testObs <- rnorm(nrow(iris)) > 0
testIris <- iris[testObs,]
trainIris <- iris[!testObs,]
multiLogit <- rxLogisticRegression(
  formula = Species~Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
  type = "multiClass", data = trainIris)

# Score the model
scoreMultiDF <- rxPredict(multiLogit, data = testIris, 
                          extraVarsToWrite = "Species")    
# Print the first rows of the data frame with scores
head(scoreMultiDF)
# Look at confusion matrix
table(scoreMultiDF$Species, scoreMultiDF$PredictedLabel)

# Look at the observations with incorrect predictions
badPrediction = scoreMultiDF$Species != scoreMultiDF$PredictedLabel
scoreMultiDF[badPrediction,]


