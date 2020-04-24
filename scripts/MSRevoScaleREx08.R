# Generated files by MSRevoScaleREx08.R placed below can be used in these scripts
# https://github.com/Azure/LearnAnalytics-AnalyzingBigDataWithMRS/tree/master/student_resources/scripts
# http://www.andresmh.com/nyctaxitrips/
# https://github.com/chendaniely/pandas_for_everyone/blob/master/data/raw_data_urls.txt
# files created by this script
# january-june 2016 - half year sample of above 60mln rows
# http://www.mediafire.com/file/pm88hgqul64p9me/yellow_tripdata_2016.xdf/file
# june 2026 - one month sample of 11135470 rows
# http://www.mediafire.com/file/q12fkyko059uies/yellow_tripdata_2016-06.xdf/file
# january-december 2015 - the whole year sample
# http://www.mediafire.com/file/rzle0hm11onl101/yellow_tripdata_2015.xdf/file

pkglist<-c("dplyr", "lubridate", "stringr", "tidyr", "rgeos", 
           "maptools", "ggplot2", "ggrepel", "ggmap", "gridExtra", 
           "evaluate", "formatR", "highr", "markdown", "yaml", 
           "htmltools", "caTools", "knitr", "rmarkdown", "lubridate", 
           "stringr")
pkgcheck <- pkglist %in% row.names(installed.packages())
pkglist[!pkgcheck]
#COMMMENT the line below if you installed packages earlier e.g on root
#for(i in pkglist[!pkgcheck]){install.packages(i,depend=TRUE)}
#this command is for root instalation of missing packages:
if(length(pkglist[!pkgcheck])) cat("install.packages(c(");j=0; for(i in pkglist[!pkgcheck]) { j<-j+1 ;  if(j == length(pkglist[!pkgcheck])) cat(paste('"',i,'"',sep="")) else cat(paste('"',i,'",',sep=""));} ; cat("),depend=TRUE)")

#loading all libraries - necessary to avoid errors of execution
for(i in pkglist) library(i, character.only = TRUE);
 

rxPrivacyControl(FALSE)
options(encoding = "UTF-8"); par(ask=F)

# Load the MicrosoftML library
library(MicrosoftML)

# in transform func you set labels: as.factor(dataList$x, levels = c(1, 2,3), 
#                           labels = c("One", "Two", "Three")) 

data_dir <- "data"
output_xdf <- file.path(data_dir, 'yellow_tripdata_2016.xdf')
my_xdf <- file.path(data_dir, 'yellow_tripdata_2016_smaller.xdf')
library(lubridate)
begin_data <- ymd("2015-12-01") # the start date - one month

col_classes <- c(
  'vendor_id'             = "integer",
  'pickup_datetime'       = "character",
  'dropoff_datetime'      = "character",
  'passenger_count'       = "integer",
  'trip_distance'         = "numeric",
  'pickup_longitude'      = "numeric",
  'pickup_latitude'       = "numeric",
  'rate_code_id'          = "factor",
  'store_and_fwd_flag'    = "character",
  'dropoff_longitude'     = "numeric",
  'dropoff_latitude'      = "numeric",
  'payment_type'          = "factor",
  'fare_amount'           = "numeric",
  'extra'                 = "numeric",
  'mta_tax'               = "numeric",
  'tip_amount'            = "numeric",
  'tolls_amount'          = "numeric",
  'improvement_surcharge' = "numeric",
  'total_amount'          = "numeric")


col_names <- c(
  'vendor_id'            ,
  'pickup_datetime'      ,
  'dropoff_datetime'     ,
  'passenger_count'      ,
  'trip_distance'        ,
  'pickup_longitude'     ,
  'pickup_latitude'      ,
  'rate_code_id'         ,
  'store_and_fwd_flag'   ,
  'dropoff_longitude'    ,
  'dropoff_latitude'     ,
  'payment_type'         ,
  'fare_amount'          ,
  'extra'                ,
  'mta_tax'              ,
  'tip_amount'           ,
  'tolls_amount'         ,
  'improvement_surcharge',
  'total_amount'        )

# generating xdf for every month csv
rxOptions(xdfCompressionLevel = 3)
# because we keep appending to the same file, we can't run this in parallel
st <- Sys.time()
ii=1
for(ii in 1:6) { # get each month's data and append it to the first month's data
  file_date <- begin_data + months(ii)
  input_csv <- sprintf('yellow_tripdata_%s.csv', substr(file_date, 1, 7))
  input_csv <- file.path(data_dir, input_csv)
  input_xdf <- sprintf('yellow_tripdata_%s.xdf', substr(file_date, 1, 7))
  input_xdf <- file.path(data_dir, input_xdf)
  dataout <- rxImport(input_csv, outFile=input_xdf, colClasses = col_classes, overwrite = TRUE)
  rxGetInfo(dataout, getVarInfo=TRUE)
  print(input_csv)
}
Sys.time() - st # stores the time it took to import

# joining months with the greatest compression about 40min for 6 month
rxOptions(xdfCompressionLevel = 9)
st <- Sys.time()
ii=7
for(ii in 1:6) { # get each month's data and append it to the first month's data
  file_date <- begin_data + months(ii)
  input_xdf <- sprintf('yellow_tripdata_%s.xdf', substr(file_date, 1, 7))
  input_xdf <- file.path(data_dir, input_xdf)
  nyc_xdf <- RxXdfData(input_xdf)
  names(nyc_xdf) <- col_names
  append <- if (ii == 1) "none" else "rows"
  #append="rows"
  dataout <- rxImport(nyc_xdf, outFile=output_xdf, 
                      colClasses = col_classes, overwrite = TRUE, append = append)
  rxGetInfo(dataout, getVarInfo=TRUE)
  print(input_xdf)
}
Sys.time() - st # stores the time it took to import

#Recompress file to smaller size (highest compression = 9)
# st <- Sys.time()
# rxCompressXdf(output_xdf, xdfCompressionLevel = 9, outFile=my_xdf)
# Sys.time() - st # stores the time it took to import

nyc_xdf <- RxXdfData(output_xdf)
rxGetInfo(nyc_xdf, getVarInfo=TRUE)

system.time(rxsum_xdf <- rxSummary( ~ fare_amount, nyc_xdf))
rxsum_xdf

###############################################################################################
# creating sample of one month taxi routes

begin_data <- ymd("2016-05-01") # the start date - one month
rxOptions(xdfCompressionLevel = 9)
# because we keep appending to the same file, we can't run this in parallel
st <- Sys.time()
ii=1
for(ii in 1:1) { # get each month's data and append it to the first month's data
  file_date <- begin_data + months(ii)
  input_csv <- sprintf('yellow_tripdata_%s.csv', substr(file_date, 1, 7))
  input_csv <- file.path(data_dir, input_csv)
  input_xdf <- sprintf('yellow_tripdata_%s.xdf', substr(file_date, 1, 7))
  input_xdf <- file.path(data_dir, input_xdf)
  dataout <- rxImport(input_csv, outFile=input_xdf, colClasses = col_classes, overwrite = TRUE)
  rxGetInfo(dataout, getVarInfo=TRUE)
  print(input_csv)
}
Sys.time() - st # stores the time it took to import

output_xdf <- file.path(data_dir, 'yellow_tripdata_2016-06-renamecol.xdf')
rxOptions(xdfCompressionLevel = 9)
st <- Sys.time()
ii=1
for(ii in 1:1) { # get each month's data and append it to the first month's data
  file_date <- begin_data + months(ii)
  input_xdf <- sprintf('yellow_tripdata_%s.xdf', substr(file_date, 1, 7))
  input_xdf <- file.path(data_dir, input_xdf)
  nyc_xdf <- RxXdfData(input_xdf)
  names(nyc_xdf) <- col_names
  append <- if (ii == 1) "none" else "rows"
  #append="rows"
  dataout <- rxImport(nyc_xdf, outFile=output_xdf, 
                      colClasses = col_classes, overwrite = TRUE, append = append)
  rxGetInfo(dataout, getVarInfo=TRUE)
  print(input_xdf)
}
Sys.time() - st # stores the time it took to import