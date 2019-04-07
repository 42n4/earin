rxPrivacyControl(FALSE)
options(encoding = "UTF-8"); par(ask=F)
# Load the MicrosoftML library
library(MicrosoftML)

#2. Load the training dataset

#Note that we have some extra code here to get the datasets directly from UCI (online) or local (offline).

unZip <- TRUE # Set to FALSE if reading directly from the trainFile and testFile
trainFile = "sentiment labelled sentences/imdb_labelled.txt"
testFile = "sentiment labelled sentences/yelp_labelled.txt"
getData <- function(targetFile, zipfile){
  read.delim(unz(zipfile, targetFile), 
             header=FALSE, 
             col.names=c("Text", "Rating"),
             quote="",
             stringsAsFactors=FALSE)
}
dataTrain <- NULL
  
if (unZip) {
  # So get a local temp location
  temp <- tempfile()
  
  # Download the zip file to the temp location
  zipfile <- download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip",temp)
  
  # We'll use the imdb_labelled.txt file for training
  dataTrain <- getData(trainFile, temp)
  dataTest <- getData(testFile, temp)
} else {
  dataTrain <- getData(trainFile)
  dataTest <- getData(testFile)
}

#3. Setup the featurizeText() transform

#For starters, we'll use the featurizeText() transform with defaults. Feel free to play around with the different transform parameters to see if you can get a better AUC result. To find out the parameters of the transform, use ?featurizeText in your R interpreter window.

#The vars parameter tells the transform which column(s) to featurzie on. In this case, it's the Text column.

# Now let's setup the text featurizer transform
# We'll use all defaults for now
textTransform = list(featurizeText(vars = c(Features = "Text")))

#4. Train a model using rxFastLinear

#Here we indicate to the rxFastLinear algorithm to use the text featurizer via the textTransform variable that was setup in Step 3. We'll use all default parameters for the rxFastLinear algorithm.

# Train a linear model on featurized text
# Using all defaults for rxFastLinear algo
model <- rxFastLinear(
  Rating ~ Features, 
  data = dataTrain,
  mlTransforms = textTransform, verbose=4
)

#5. Test the model using the test dataset

#Now let's test the model. We'll use yelp_labelled.txt as the test dataset.

# Get the predictions based on the test dataset
score <- rxPredict(model, data = dataTest, extraVarsToWrite = c("Rating"))

# Let's look at the prediction
head(score)

#6. Evaluate the performance of the model

#Finally, let's see how good the model was based on the test dataset.

# How good was the prediction?
rxRocCurve(actualVarName = "Rating", predVarNames = "Probability.1", data = score)
