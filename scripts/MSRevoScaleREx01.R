#################################################################################
#https://docs.microsoft.com/en-us/machine-learning-server/
rxPrivacyControl(FALSE)
options(encoding = "UTF-8"); par(ask=F)
library(MicrosoftML)
getwd()

packageVersion("RevoScaleR")
ls("package:RevoScaleR")
find("rxKmeans")
ls("package:MicrosoftML")

#wget https://packages.revolutionanalytics.com/datasets/AirlineDataSubsample.xdf
#airdelay<-rxReadXdf("AirlineDataSubsample.xdf")
#rxGetInfo(airdelay, getVarInfo=TRUE)

#####################################################################################
#https://docs.microsoft.com/en-us/machine-learning-server/r/tutorial-revoscaler-data-import-transform
#####################################################################################
##normal packages data samples
# data()
##M$ packages data samples
# list.files(rxGetOption("sampleDataDir"))

#airlineSmall <- file.path(rxGetOption("sampleDataDir"), "AirlineDemoSmall.xdf")
mysource <- file.path(rxGetOption("sampleDataDir"), "AirlineDemoSmall.csv")
airXdfData <- rxImport(inData=mysource)
airXdfData <- rxImport(inData=mysource, outFile="data/airExample.xdf",overwrite=TRUE)
rxGetInfo(airXdfData, getVarInfo = TRUE)

airXdfData <- rxImport(inData=mysource, outFile="data/airExample.xdf",
                       stringsAsFactors=TRUE, missingValueString="M", rowsPerRead=200000,
                       overwrite=TRUE)

colInfo <- list(DayOfWeek = list(type = "factor",
                                 levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")))

airXdfData <- rxImport(inData=mysource, outFile="data/airExample.xdf", missingValueString="M", 
                       rowsPerRead=200000, colInfo  = colInfo, colClasses=c(ArrDelay="integer"),
                       overwrite=TRUE)


rxHistogram(~ArrDelay|DayOfWeek,  data = airXdfData)
rxSummary(~ ArrDelay, data = airXdfData)

#adding a new row
airXdfData <- rxDataStep(inData = airXdfData, outFile = "data/airExample.xdf",
                         transforms=list(VeryLate = (ArrDelay > 120 | is.na(ArrDelay))),
                         overwrite = TRUE)
rxGetInfo(airXdfData, getVarInfo = TRUE)
rxSummary(~ArrDelay+CRSDepTime+DayOfWeek, data=airXdfData)
summary(airXdfData)
options("device.ask.default" = T)
rxHistogram(~ArrDelay, data=airXdfData)
rxHistogram(~CRSDepTime, data=airXdfData)
rxHistogram(~DayOfWeek, data=airXdfData)

myData <- rxDataStep(inData = airXdfData,
                     rowSelection = ArrDelay > 240 & ArrDelay <= 300,
                     varsToKeep = c("ArrDelay", "DayOfWeek"),overwrite = TRUE)
rxHistogram(~ArrDelay, data = myData)

nrow(airXdfData)
ncol(airXdfData)
head(airXdfData)

airXdfDataSmall <- rxDataStep(inData=airXdfData, numRows=10, startRow=100000)
airXdfDataSmall

airExtraDS <- rxDataStep(inData=airXdfData, outFile="data/ADS2.xdf",
                         transforms=list(
                           Late = ArrDelay > 15,
                           DepHour = as.integer(CRSDepTime),
                           Night = DepHour >= 20 | DepHour <= 5),overwrite = TRUE)

rxGetInfo(airExtraDS, getVarInfo=TRUE, numRows=5)


#####################################################################################
#https://docs.microsoft.com/en-us/machine-learning-server/r/tutorial-revoscaler-large-data-airline
#####################################################################################

bigDataDir <- "data"
#wget https://packages.revolutionanalytics.com/datasets/AirOnTime87to12/AirOnTime87to12.xdf -O data/AirOnTime87to12.xdf
airDataName <- file.path(bigDataDir, "AirOnTime87to12.xdf")
bigAirDS <- RxXdfData( airDataName )
rxGetInfo(bigAirDS, getVarInfo=TRUE)

#normal lm on a small data
testDF <- rxDataStep(inData = bigAirDS,
                     varsToKeep = c("ArrDelay","DepDelay", "DayOfWeek"),
                     startRow = 100000, numRows = 1000)
summary(testDF)
lmObj <- lm(ArrDelay~DayOfWeek, data = testDF)
summary(lmObj)

#Big Data read and lm
system.time(
  delayArr <- rxLinMod(ArrDelay ~ DayOfWeek, data = bigAirDS,
                       cube = TRUE, blocksPerRead = 30)
)

summary(delayArr)

delayDep <- rxLinMod(DepDelay ~ DayOfWeek, data = bigAirDS,
                     cube = TRUE, blocksPerRead = 30)

cubeResults <- rxResultsDF(delayArr)
cubeResults$DepDelay <- rxResultsDF(delayDep)$DepDelay

rxLinePlot( ArrDelay + DepDelay ~ DayOfWeek, data = cubeResults,
            title = 'Average Arrival and Departure Delay by Day of Week')

delayCarrier <- rxLinMod(ArrDelay ~ UniqueCarrier,
                         data = bigAirDS, cube = TRUE, blocksPerRead = 30)

dcCoef <- sort(coef(delayCarrier))

head(dcCoef, 10)
tail(dcCoef, 10)

sprintf("United's additional delay compared with Hawaiian: %f",
        dcCoef["UniqueCarrier=UA"]-dcCoef["UniqueCarrier=HA"])


delayCarrierLoc <- rxLinMod(ArrDelay ~ UniqueCarrier + Origin+Dest,
                            data = bigAirDS, cube = TRUE, blocksPerRead = 30)

dclCoef <- coef(delayCarrierLoc)
sprintf(
  "United's additional delay accounting for dep and arr location: %f",
  dclCoef["UniqueCarrier=UA"]- dclCoef["UniqueCarrier=HA"])
paste("Number of coefficients estimated: ", length(!is.na(dclCoef)))

delayCarrierLocHour <- rxLinMod(ArrDelay ~
                                  UniqueCarrier + Origin + Dest + F(CRSDepTime),
                                data = bigAirDS, cube = TRUE, blocksPerRead = 30)
dclhCoef <- coef(delayCarrierLocHour)

#Now we can summarize all of our results:
  
  sprintf("United's additional delay compared with Hawaiian: %f",
          dcCoef["UniqueCarrier=UA"]-dcCoef["UniqueCarrier=HA"])
paste("Number of coefficients estimated: ", length(!is.na(dcCoef)))
sprintf(
  "United's additional delay accounting for dep and arr location: %f",
  dclCoef["UniqueCarrier=UA"]- dclCoef["UniqueCarrier=HA"])
paste("Number of coefficients estimated: ", length(!is.na(dclCoef)))
sprintf(
  "United's additional delay accounting for location and time: %f",    
  dclhCoef["UniqueCarrier=UA"]-dclhCoef["UniqueCarrier=HA"])
paste("Number of coefficients estimated: ", length(!is.na(dclhCoef)))

expectedDelay <- function( carrier = "AA", origin = "SEA",
                           dest = "SFO", deptime = "9")
{
  coeffNames <- c(
    sprintf("UniqueCarrier=%s", carrier),
    sprintf("Origin=%s", origin),
    sprintf("Dest=%s", dest),
    sprintf("F_CRSDepTime=%s", deptime))
  return (sum(dclhCoef[coeffNames]))
}

#We can use this function to compare, for example, the expected delay for trips going from Seattle to New York, or Newark, or Honolulu:
  
# Go to JFK (New York) from Seattle at 5 in the afternoon on United
expectedDelay("AA", "SEA", "JFK", "17")
# Go to Newark from Seattle at 5 in the afternoon on United
expectedDelay("UA", "SEA", "EWR", "17")
# Or go to Honolulu from Seattle at 7 am on Hawaiian
expectedDelay("HA", "SEA", "HNL", "7")

#####################################################################################
#https://docs.microsoft.com/en-us/machine-learning-server/r/how-to-developer-write-chunking-algorithms
#####################################################################################

chunkTable <- function(inDataSource, iroDataSource, varsToKeep = NULL,
                       blocksPerRead = 1 )
{
  ProcessChunk <- function( dataList)
  {
    # Process Data
    chunkTable <- table(as.data.frame(dataList))
    # Convert table to data frame with single row
    varNames <- names(chunkTable)
    varValues <- as.vector(chunkTable)
    dim(varValues) <- c(1, length(varNames))
    chunkDF <- as.data.frame(varValues)
    names(chunkDF) <- varNames
    # Return the data frame
    return( chunkDF )
  }
  
  rxDataStep( inData = inDataSource, outFile = iroDataSource,
              varsToKeep = varsToKeep,
              blocksPerRead = blocksPerRead,
              transformFunc = ProcessChunk,
              reportProgress = 0, overwrite = TRUE)
  
  AggregateResults <- function()    
  {
    iroResults <- rxDataStep(iroDataSource)
    return(colSums(iroResults))
  }
  
  return(AggregateResults())
}

inDataSource <- file.path(rxGetOption("sampleDataDir"),
                          "AirlineDemoSmall.xdf")
iroDataSource <- "iroFile.xdf"
chunkOut <- chunkTable(inDataSource = inDataSource,
                       iroDataSource = iroDataSource, varsToKeep="DayOfWeek")
chunkOut

rxDataStep(iroDataSource)
file.remove(iroDataSource)

file.remove("data/airExample.xdf")
file.remove("data/ADS2.xdf")

#Further steps
#/opt/microsoft/mlserver/9.3.0/libraries/RServer/RevoScaleR/demoScripts/RevoScaleR_Getting_Started.R
#/opt/microsoft/mlserver/9.3.0/libraries/RServer/RevoScaleR/demoScripts/AirOnTime2012Analysis.R
#/opt/microsoft/mlserver/9.3.0/libraries/RServer/RevoScaleR/demoScripts/AirOnTime87to12Import.R
#/opt/microsoft/mlserver/9.3.0/libraries/RServer/RevoScaleR/demoScripts/RevoScaleR_Users_Guide.R


#Other tutorials
#https://docs.microsoft.com/en-us/machine-learning-server/r/tutorial-revoscaler-data-model-analysis
#https://docs.microsoft.com/en-us/machine-learning-server/r/tutorial-revoscaler-large-data-airline
#https://docs.microsoft.com/en-us/machine-learning-server/r/tutorial-revoscaler-large-data-census
#https://docs.microsoft.com/en-us/machine-learning-server/r/tutorial-revoscaler-large-data-loan
#https://docs.microsoft.com/en-us/sql/advanced-analytics/tutorials/deepdive-data-science-deep-dive-using-the-revoscaler-packages?view=sql-server-2017
#https://github.com/mmparker-msft/intro_to_mrs
#https://github.com/Microsoft/MRSWorkshopStudent/tree/master/Labs
