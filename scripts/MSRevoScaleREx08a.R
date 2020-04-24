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
#                            labels = c("One", "Two", "Three")) 

data_dir <- "data"
output_xdf <- file.path(data_dir, 'yellow_tripdata_2016-06.xdf')
my_xdf <- file.path(data_dir, 'yellow_tripdata_2016_smaller.xdf')
library(lubridate)
begin_data <- ymd("2015-12-01") # the day of the months is irrelevant

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

nyc_xdf <- RxXdfData(output_xdf)
rxGetInfo(nyc_xdf, getVarInfo=TRUE)
rxOptions(xdfCompressionLevel = 1) # for quickest operations the lowest compression

###################################################################################
# PREPARING & TESTING THE DATA
###################################################################################

## ----chap03chunk02-------------------------------------------------------
rxGetInfo(nyc_xdf, getVarInfo = TRUE, numRows = 5)

## ----chap03chunk03-------------------------------------------------------
rxDataStep(nyc_xdf, nyc_xdf,
  transforms = list(
    tip_percent = ifelse(fare_amount > 0 & tip_amount < fare_amount, 
                         round(tip_amount * 100 / fare_amount, 0), 
                         NA)),
  overwrite = TRUE)
rxSummary( ~ tip_percent, nyc_xdf)

## ----chap03chunk04-------------------------------------------------------
rxSummary( ~ tip_percent2, nyc_xdf,
           transforms = list(
            tip_percent2 = ifelse(fare_amount > 0 & tip_amount < fare_amount, 
                                  round(tip_amount * 100 / fare_amount, 0), 
                                  NA)))

## ----chap03chunk05-------------------------------------------------------
rxCrossTabs( ~ month:year, nyc_xdf,
             transforms = list(
               year = as.integer(substr(pickup_datetime, 1, 4)),
               month = as.integer(substr(pickup_datetime, 6, 7)),
               year = factor(year, levels = 2014:2016),
               month = factor(month, levels = 1:12)))

## ----chap03chunk06-------------------------------------------------------
rxCrossTabs( ~ month:year, nyc_xdf,
             transforms = list(
               date = ymd_hms(pickup_datetime),
               year = factor(year(date), levels = 2014:2016),
               month = factor(month(date), levels = 1:12)),
             transformPackages = "lubridate")

## ----chap03chunk07-------------------------------------------------------
# transformation function for extracting some date and time features
xforms <- function(data) {
  wlabels <- c('Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat')
  cut_levels <- c(1, 5, 9, 12, 16, 18, 22)
  hour_labels <- c('1AM-5AM', '5AM-9AM', '9AM-12PM', '12PM-4PM', 
                   '4PM-6PM', '6PM-10PM', '10PM-1AM')
  
  pickup_datetime <- ymd_hms(data$pickup_datetime, tz = "UTC")
  pickup_hour <- addNA(cut(hour(pickup_datetime), cut_levels))
  pickup_dow <- factor(wday(pickup_datetime), levels = 1:7, labels = wlabels)
  levels(pickup_hour) <- hour_labels
  
  dropoff_datetime <- ymd_hms(data$dropoff_datetime, tz = "UTC")
  dropoff_hour <- addNA(cut(hour(dropoff_datetime), cut_levels))
  dropoff_dow <- factor(wday(dropoff_datetime), levels = 1:7, labels = wlabels)
  levels(dropoff_hour) <- hour_labels
  
  data$pickup_hour <- pickup_hour
  data$pickup_dow <- pickup_dow
  data$dropoff_hour <- dropoff_hour
  data$dropoff_dow <- dropoff_dow
  data$trip_duration <- as.integer(as.duration(dropoff_datetime - pickup_datetime))
  
  data
}

# sample_xdf <- file.path(data_dir, 'yellow_tripdata_2016-06.xdf')
# nyc_sample <- RxXdfData(sample_xdf)
# rxGetInfo(nyc_sample, getVarInfo=TRUE)
# rxOptions(xdfCompressionLevel = 1) # for quickest operations the lowest compression

## ----chap03chunk08-------------------------------------------------------
library(lubridate)
# Sys.setenv(TZ = "US/Eastern") # not important for this dataset
# xforms(head(nyc_sample)) # test the function on a data.frame
# 
# ## ----chap03chunk09-------------------------------------------------------
# head(rxDataStep(nyc_sample, transformFunc = xforms, transformPackages = "lubridate"))

## ----chap03chunk10-------------------------------------------------------
st <- Sys.time()
rxDataStep(nyc_xdf, nyc_xdf, transformFunc = xforms, overwrite = TRUE, 
           transformPackages = "lubridate")
Sys.time() - st

## ----chap03chunk11-------------------------------------------------------
rxs1 <- rxSummary( ~ pickup_hour + pickup_dow + trip_duration, nyc_xdf)
# we can add a column for proportions next to the counts
rxs1$categorical <- lapply(rxs1$categorical, 
  function(x) cbind(x, prop = round(prop.table(x$Counts), 2)))
rxs1

## ----chap03chunk12-------------------------------------------------------
rxs2 <- rxSummary( ~ pickup_dow:pickup_hour, nyc_xdf)
rxs2 <- tidyr::spread(rxs2$categorical[[1]], key = 'pickup_hour', value = 'Counts')
row.names(rxs2) <- rxs2[ , 1]
rxs2 <- as.matrix(rxs2[ , -1])
rxs2

## ----chap03chunk13-------------------------------------------------------
levelplot(prop.table(rxs2, 2), cuts = 10, xlab = "", ylab = "", 
          main = "Distribution of taxis by day of week")

## ----chap03chunk14-------------------------------------------------------
library(rgeos)
library(maptools)

nyc_shapefile <- readShapePoly(file.path(data_dir, 'ZillowNeighborhoods-NY/ZillowNeighborhoods-NY.shp'))
mht_shapefile <- subset(nyc_shapefile, City == 'New York' & County == 'New York')

mht_shapefile@data$id <- as.character(mht_shapefile@data$Name)
library(ggplot2)
mht_points <- fortify(gBuffer(mht_shapefile, byid = TRUE, width = 0), region = "Name")
library(dplyr)
mht_df <- inner_join(mht_points, mht_shapefile@data, by = "id")
mht_cent <- cbind(mht_shapefile@data, as.data.frame(gCentroid(mht_shapefile, byid = TRUE)))

library(ggrepel)
ggplot(mht_df, aes(long, lat, fill = id)) +
  geom_polygon() +
  geom_path(color = "white") +
  coord_equal() +
  theme(legend.position = "none") +
  geom_text_repel(aes(label = id, x = x, y = y), data = mht_cent, size = 3)

# ## ----chap03chunk15-------------------------------------------------------
# # take only the coordinate columns, and replace NAs with 0
# data_coords_fun <- function(data) {
#   data$long = ifelse(is.na(data$pickup_longitude), 0, data$pickup_longitude)
#   data$lat  = ifelse(is.na(data$pickup_latitude),  0, data$pickup_latitude)
# }
# 
# rxDataStep(nyc_sample, nyc_sample, transformFunc = data_coords_fun, transformPackages = "lubridate")
# 
# # we specify the columns that correspond to the coordinates
# coordinates(data_coords) <- c('long', 'lat')
# # returns the neighborhoods based on coordinates
# nhoods <- over(data_coords, nyc_shapefile)
# # rename the column names in nhoods
# names(nhoods) <- paste('pickup', tolower(names(nhoods)), sep = '_')
# # combine the neighborhood information with the original data
# nyc_sample <- cbind(nyc_sample, nhoods)
# head(nyc_sample)

## ----chap03chunk16, eval=FALSE-------------------------------------------
## find_nhoods <- function(data) {
##   # extract pick-up lat and long and find their neighborhoods
##   # add only the pick-up neighborhood and city columns to the data
##   # extract drop-off lat and long and find their neighborhoods
##   # add only the drop-off neighborhood and city columns to the data
##   # return the data with the new columns added in
## }

## ----chap03chunk17, eval=FALSE-------------------------------------------
## # test the function on a data.frame using rxDataStep
## head(rxDataStep(nyc_sample,
##                 transformFunc = find_nhoods,
##                 transformPackages = c("sp", "maptools"),
##                 transformObjects = list(shapefile = nyc_shapefile)))

## ----chap03chunk18-------------------------------------------------------
rxDataStep(nyc_xdf, nyc_xdf,
  transforms = list(
    rate_code_id = as.factor(rate_code_id),
    rate_code_id = factor(rate_code_id, 
                          levels = 1:6, 
                          labels = c('standard', 'JFK', 'Newark', 'Nassau or Westchester', 
                                     'negotiated', 'group ride')),
    payment_type = factor(payment_type, levels = 1:2, labels = c('card', 'cash'))
  ),
  overwrite = TRUE)

## ----chap03chunk19-------------------------------------------------------
nhoods <- over(data_coords, mht_shapefile)
str(nhoods)

## ----chap03chunk20-------------------------------------------------------
find_nhoods <- function(data) {
  # extract pick-up lat and long and find their neighborhoods
  pickup_longitude <- ifelse(is.na(data$pickup_longitude), 0, data$pickup_longitude)
  pickup_latitude <- ifelse(is.na(data$pickup_latitude), 0, data$pickup_latitude)
  data_coords <- data.frame(long = pickup_longitude, lat = pickup_latitude)
  coordinates(data_coords) <- c('long', 'lat')
  nhoods <- over(data_coords, shapefile)
  # add only the pick-up neighborhood and city columns to the data
  data$pickup_nhood <- nhoods$Name
  data$pickup_borough <- nhoods$County
  # extract drop-off lat and long and find their neighborhoods
  dropoff_longitude <- ifelse(is.na(data$dropoff_longitude), 0, data$dropoff_longitude)
  dropoff_latitude <- ifelse(is.na(data$dropoff_latitude), 0, data$dropoff_latitude)
  data_coords <- data.frame(long = dropoff_longitude, lat = dropoff_latitude)
  coordinates(data_coords) <- c('long', 'lat')
  nhoods <- over(data_coords, shapefile)
  # add only the drop-off neighborhood and city columns to the data
  data$dropoff_nhood <- nhoods$Name
  data$dropoff_borough <- nhoods$County
  # return the data with the new columns added in
  data
}

## ----chap03chunk21-------------------------------------------------------
# test the function on a data.frame using rxDataStep
# head(rxDataStep(nyc_sample, transformFunc = find_nhoods, 
#                 transformPackages = c("sp", "maptools"), 
#                 transformObjects = list(shapefile = nyc_shapefile)))

## ----chap03chunk22-------------------------------------------------------
st <- Sys.time()
rxDataStep(nyc_xdf, nyc_xdf, overwrite = TRUE, 
           transformFunc = find_nhoods, 
           transformPackages = c("sp", "maptools", "rgeos"), 
           transformObjects = list(shapefile = nyc_shapefile))

Sys.time() - st
rxGetInfo(nyc_xdf, numRows = 5)

## ----chap04chunk02-------------------------------------------------------
system.time(rxs_all <- rxSummary( ~ ., nyc_xdf) )

## ----chap04chunk03-------------------------------------------------------
head(rxs_all$sDataFrame)

## ----chap04chunk04-------------------------------------------------------
nhoods_by_borough <- rxCube( ~ pickup_nhood:pickup_borough, nyc_xdf, returnDataFrame = TRUE)
library(dplyr)
nhoods_by_borough %>%
  select(pickup_borough, pickup_nhood, Counts) %>%
  filter(Counts > 0) %>%
  arrange(pickup_borough, desc(Counts)) %>%
  group_by(pickup_borough) %>%
  top_n(5)

## ----chap04chunk05-------------------------------------------------------
nhoods_by_borough <- rxCrossTabs( ~ pickup_nhood:pickup_borough, nyc_xdf)
nhoods_by_borough <- nhoods_by_borough$counts[[1]]
nb_cnt <- apply(nhoods_by_borough, 1, function(x) sum(x > 0))
nb_cnt[nb_cnt > 1]

## ----chap04chunk06-------------------------------------------------------
manhattan_nhoods <- subset(nyc_shapefile@data, County == 'New York', select = "Name", drop = TRUE)
# manhattan_nhoods <- manhattan_nhoods[-grep('Island', manhattan_nhoods)]
manhattan_nhoods <- as.character(manhattan_nhoods)
bad_nhoods <- c('Brooklyn Heights', 'Marble Hill', 'Ellis Island', 'Liberty Island',
                'Mill Rock Park', 'Governors Island', 'Vinegar Hill')
manhattan_nhoods <- setdiff(manhattan_nhoods, bad_nhoods)

refactor_columns <- function(data) {
  data$pickup_nb = factor(data$pickup_nhood, levels = nhoods_levels)
  data$dropoff_nb = factor(data$dropoff_nhood, levels = nhoods_levels)
  data
}

rxDataStep(nyc_xdf, nyc_xdf, overwrite = TRUE, 
           transformFunc = refactor_columns, 
           transformObjects = list(nhoods_levels = manhattan_nhoods))

rxs_pickdrop <- rxSummary( ~ pickup_nb:dropoff_nb, nyc_xdf)
head(rxs_pickdrop$categorical[[1]])

## ----chap04chunk07-------------------------------------------------------
rxHistogram( ~ trip_distance, nyc_xdf, startVal = 0, endVal = 25, 
             histType = "Percent", numBreaks = 20)

## ----chap04chunk08-------------------------------------------------------
rxs <- rxSummary( ~ pickup_nhood:dropoff_nhood, nyc_xdf, 
                  rowSelection = (trip_distance > 15 & trip_distance < 22))
library(dplyr)
head(arrange(rxs$categorical[[1]], desc(Counts)), 10)

## ----chap04chunk09-------------------------------------------------------
# outFile argument missing means we output to data.frame
odd_trips <- rxDataStep(nyc_xdf, 
                        rowSelection = (u < .05 & ( # we can adjust this if the data gets too big
                          (trip_distance > 20 | trip_distance <= 0) |
                            (passenger_count > 5 | passenger_count == 0) |
                            (fare_amount > 1000 | fare_amount <= 0))), 
                        transforms = list(u = runif(.rxNumRows)))

print(dim(odd_trips))

## ----chap04chunk10-------------------------------------------------------
library(ggplot2)
odd_trips %>%
  filter(trip_distance > 20) %>%
  ggplot() -> p

p + geom_histogram(aes(x = fare_amount, fill = trip_duration <= 10*60), binwidth = 10) +
  xlim(0, 500) + 
  coord_fixed(ratio = .5)

## ----chap04chunk11-------------------------------------------------------
rxHistogram( ~ trip_distance, nyc_xdf, startVal = 0, endVal = 25, 
             histType = "Percent", numBreaks = 20)

## ----chap04chunk12-------------------------------------------------------
cut(8.9, breaks = c(-Inf, 0, 5, 10, Inf), labels = c("0", "<5", "5-10", "10+"))

## ----chap04chunk13-------------------------------------------------------
rxHistogram( ~ trip_distance | pickup_hour + payment_type, nyc_xdf, startVal = 0, 
             endVal = 25, numBreaks = 20)

## ----chap04chunk14-------------------------------------------------------
rxHistogram( ~ trip_dist | pickup_hour + payment_type, nyc_xdf, 
             transforms = list(trip_dist = cut(trip_distance, 
                                               breaks = c(-Inf, 0, 5, 10, Inf), 
                                               labels = c("0", "<5", "5-10", "10+"))))


###################################################################################
# CLUSTERING RXKMEANS KMEANS
###################################################################################

## ----chap05chunk02-------------------------------------------------------
input_xdf <- file.path(data_dir, 'yellow_tripdata_2016_clean.xdf')
mht_xdf <- RxXdfData(input_xdf)

rxDataStep(nyc_xdf, mht_xdf,
           rowSelection = (passenger_count > 0 &
                             trip_distance >= 0 & trip_distance < 30 &
                             trip_duration > 0 & trip_duration < 60*60*24 &
                             !is.na(pickup_nb) &
                             !is.na(dropoff_nb) &
                             fare_amount > 0),
           transformPackages = "stringr",
           varsToDrop = c('extra', 'mta_tax', 'improvement_surcharge', 'total_amount', 
                          'pickup_borough', 'dropoff_borough', 'pickup_nhood', 'dropoff_nhood'),
           overwrite = TRUE)
rxGetInfo(mht_xdf, getVarInfo=TRUE)
## ----chap05chunk03-------------------------------------------------------
mht_sample <- rxDataStep(mht_xdf, rowSelection = (u < .01), transforms = list(u = runif(.rxNumRows)))

dim(mht_sample)

## ----chap05chunk04-------------------------------------------------------
library(ggmap)

# Use your own google maps key (you can get it free with 200$ account)
# https://console.cloud.google.com/google/maps-apis/overview
# this key is a fake from R help
# register_google(key = "mQkzTpiaLYjPqXQBotesgif3EfGL2dbrNVOrogg")

map_13 <- get_map(location = c(lon = -73.98, lat = 40.76), zoom = 13)
map_14 <- get_map(location = c(lon = -73.98, lat = 40.76), zoom = 14)
map_15 <- get_map(location = c(lon = -73.98, lat = 40.76), zoom = 15)

q1 <- ggmap(map_14) +
  geom_point(aes(x = dropoff_longitude, y = dropoff_latitude), data = mht_sample, 
             alpha = 0.15, na.rm = TRUE, col = "red", size = .5) +
  theme_nothing(legend = TRUE)

q2 <- ggmap(map_15) +
  geom_point(aes(x = dropoff_longitude, y = dropoff_latitude), data = mht_sample, 
             alpha = 0.15, na.rm = TRUE, col = "red", size = .5) +
  theme_nothing(legend = TRUE)

library(gridExtra)
grid.arrange(q1, q2, ncol = 2)

## ----chap05chunk05-------------------------------------------------------
library(dplyr)
xydata <- transmute(mht_sample, 
                    long_std = dropoff_longitude / -74, 
                    lat_std = dropoff_latitude / 40)

start_time <- Sys.time()
rxkm_sample <- kmeans(xydata, centers = 300, iter.max = 2000, nstart = 50)
Sys.time() - start_time

# we need to put the centroids back into the original scale for coordinates
centroids_sample <- rxkm_sample$centers %>%
  as.data.frame %>%
  transmute(long = long_std*(-74), lat = lat_std*40, size = rxkm_sample$size)

head(centroids_sample)

## ----chap05chunk06-------------------------------------------------------
start_time <- Sys.time()
rxkm <- rxKmeans( ~ long_std + lat_std, data = mht_xdf, outFile = mht_xdf, 
                  outColName = "dropoff_cluster", centers = rxkm_sample$centers, 
                  transforms = list(long_std = dropoff_longitude / -74, 
                                    lat_std = dropoff_latitude / 40), 
                  blocksPerRead = 1, overwrite = TRUE, maxIterations = 100, 
                  reportProgress = -1)
Sys.time() - start_time

clsdf <- cbind(
  transmute(as.data.frame(rxkm$centers), long = long_std*(-74), lat = lat_std*40),
  size = rxkm$size, withinss = rxkm$withinss)

head(clsdf)

## ----chap05chunk07-------------------------------------------------------
centroids_whole <- cbind(transmute(as.data.frame(rxkm$centers), 
                                   long = long_std*(-74), lat = lat_std*40), 
                         size = rxkm$size, 
                         withinss = rxkm$withinss)

q1 <- ggmap(map_15) +
  geom_point(data = centroids_sample, aes(x = long, y = lat, alpha = size), 
             na.rm = TRUE, size = 3, col = 'red') +
  theme_nothing(legend = TRUE) +
  labs(title = "centroids using sample data")

q2 <- ggmap(map_15) +
  geom_point(data = centroids_whole, aes(x = long, y = lat, alpha = size), 
             na.rm = TRUE, size = 3, col = 'red') +
  theme_nothing(legend = TRUE) +
  labs(title = "centroids using whole data")

library(gridExtra)
grid.arrange(q1, q2, ncol = 2)

## ----chap05chunk08-------------------------------------------------------
nclus <- 50
kmeans_nclus <- kmeans(xydata, centers = nclus, iter.max = 2000, nstart = 1)
sum(kmeans_nclus$withinss)

## ----chap05chunk09-------------------------------------------------------
nclus_seq <- seq(20, 1000, by = 50)

## ----chap05chunk10-------------------------------------------------------
find_wss <- function(nclus, ...) {
  st <- Sys.time()
  res <- sum(kmeans(centers = nclus, ...)$withinss)
  print(sprintf("nclus = %d, runtime = %3.2f seconds", nclus, Sys.time() - st))
  res
}

find_wss(nclus = 10, x = xydata, iter.max = 500, nstart = 1)

## ----chap05chunk11-------------------------------------------------------
wss <- sapply(nclus_seq, find_wss, x = xydata, iter.max = 500, nstart = 1)

library(ggplot2)
ggplot(aes(x = x, y = y), data = data.frame(x = nclus_seq, y = wss)) +
  geom_line() +
  xlab("number of clusters") +
  ylab("within clusters sum of squares")


###################################################################################
# NEIGHBORHOOD PATTERNS
###################################################################################

## ----chap06chunk02-------------------------------------------------------
rxct <- rxCrossTabs(trip_distance ~ pickup_nb:dropoff_nb, mht_xdf)
res <- rxct$sums$trip_distance / rxct$counts$trip_distance

library(seriation)
res[which(is.nan(res))] <- mean(res, na.rm = TRUE)
nb_order <- seriate(res)

## ----chap06chunk03-------------------------------------------------------
rxc1 <- rxCube(trip_distance ~ pickup_nb:dropoff_nb, mht_xdf)
rxc2 <- rxCube(minutes_per_mile ~ pickup_nb:dropoff_nb, mht_xdf, 
               transforms = list(minutes_per_mile = (trip_duration/60)/trip_distance))
rxc3 <- rxCube(tip_percent ~ pickup_nb:dropoff_nb, mht_xdf)
library(dplyr)
res <- dplyr::bind_cols(c(rxc1, rxc2, rxc3))
res <- res[ , c('pickup_nb', 'dropoff_nb', 'trip_distance', 'minutes_per_mile', 'tip_percent')]
head(res)

## ----chap06chunk04-------------------------------------------------------
library(ggplot2)
ggplot(res, aes(pickup_nb, dropoff_nb)) +
  geom_tile(aes(fill = trip_distance), colour = "white") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  coord_fixed(ratio = .9)

## ----chap06chunk05-------------------------------------------------------
newlevs <- levels(res$pickup_nb)[unlist(nb_order)]
res$pickup_nb <- factor(res$pickup_nb, levels = unique(newlevs))
res$dropoff_nb <- factor(res$dropoff_nb, levels = unique(newlevs))

ggplot(res, aes(pickup_nb, dropoff_nb)) +
  geom_tile(aes(fill = trip_distance), colour = "white") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  coord_fixed(ratio = .9)

## ----chap06chunk06-------------------------------------------------------
ggplot(res, aes(pickup_nb, dropoff_nb)) +
  geom_tile(aes(fill = minutes_per_mile), colour = "white") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  coord_fixed(ratio = .9)

## ----chap06chunk07-------------------------------------------------------
res %>%
  mutate(tip_color = cut(tip_percent, c(0, 8, 12, 15, 100))) %>%
  ggplot(aes(pickup_nb, dropoff_nb)) +
  geom_tile(aes(fill = tip_color)) +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  coord_fixed(ratio = .9)

## ----chap06chunk08-------------------------------------------------------
# first way of reordering the factor levels
rxDataStep(inData = mht_xdf, outFile = mht_xdf, 
           transforms = list(
             pickup_nb = factor(pickup_nb, levels = newlevels), 
             dropoff_nb = factor(dropoff_nb, levels = newlevels)), 
           transformObjects = list(newlevels = unique(newlevs)), overwrite = TRUE)

## ----chap06chunk09, eval=FALSE-------------------------------------------
## # second way of reordering the factor levels
## rxFactors(mht_xdf, outFile = mht_xdf,
## 	factorInfo = list(
## 		pickup_nb = list(newLevels = unique(newlevs)),
## 		dropoff_nb = list(newLevels = unique(newlevs))),
## 	overwrite = TRUE)

## ----chap06chunk10-------------------------------------------------------
rxc <- rxCube( ~ pickup_nb:dropoff_nb, mht_xdf)
rxc <- as.data.frame(rxc)

rxc %>%
  filter(Counts > 0) %>%
  mutate(pct_all = Counts/sum(Counts) * 100) %>%
  group_by(pickup_nb) %>%
  mutate(pct_by_pickup_nb = Counts/sum(Counts) * 100) %>%
  group_by(dropoff_nb) %>%
  mutate(pct_by_dropoff_nb = Counts/sum(Counts) * 100) %>%
  group_by() %>%
  arrange(desc(Counts)) -> rxcs

head(rxcs)

## ----chap06chunk11-------------------------------------------------------
ggplot(rxcs, aes(pickup_nb, dropoff_nb)) +
  geom_tile(aes(fill = pct_all), colour = "white") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  scale_fill_gradient(low = "white", high = "black") +
  coord_fixed(ratio = .9)

## ----chap06chunk12-------------------------------------------------------
ggplot(rxcs, aes(pickup_nb, dropoff_nb)) +
  geom_tile(aes(fill = pct_by_pickup_nb), colour = "white") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  coord_fixed(ratio = .9)

## ----chap06chunk13-------------------------------------------------------
ggplot(rxcs, aes(pickup_nb, dropoff_nb)) +
  geom_tile(aes(fill = pct_by_dropoff_nb), colour = "white") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  scale_fill_gradient(low = "white", high = "red") +
  coord_fixed(ratio = .9)

## ----chap06chunk14-------------------------------------------------------
rxcs %>%
  select(pickup_nb, dropoff_nb, pct = pct_by_pickup_nb) %>%
  arrange(pickup_nb, desc(pct))

## ----chap06chunk15, eval=FALSE-------------------------------------------
## nb_name <- "West Village" # a neighborhood of our choosing
## nb_drop <- ## pull the most common destinations for this neighborhood from `rxcs_tops`
## 
## pickup_df <- rxDataStep(mht_xdf, # we leave out outFile and store results in pickup_df
##   rowSelection = ## select the relevant subset of the data
##   varsToKeep = c("dropoff_nb", "pickup_datetime"),
##   transformObjects = ## a list, used to pass `nb_name` and `nb_drop` to rowSelection
##   )

## ----chap06chunk16, eval=FALSE-------------------------------------------
## library(lubridate)
## pickup_df %>%
##   mutate(pickup_hour = hour(ymd_hms(pickup_datetime, tz = "UTC"))) %>%
##   ggplot(aes(x = pickup_hour, fill = dropoff_nb)) +
##   geom_bar(position = "stack", stat = "count") +
##   scale_fill_discrete(guide = guide_legend(reverse = TRUE))

## ----chap06chunk17-------------------------------------------------------
rxcs %>%
  select(pickup_nb, dropoff_nb, pct = pct_by_pickup_nb) %>%
  arrange(pickup_nb, desc(pct)) %>%
  group_by(pickup_nb) %>%
  mutate(cumpct = cumsum(pct)) %>%
  filter(pct > 5 & (cumpct <= 50 | (cumpct > 50 & lag(cumpct) <= 50))) %>%
  as.data.frame -> rxcs_tops

## ----chap06chunk18-------------------------------------------------------
nb_name <- "West Village"
nb_drop <- subset(rxcs_tops, pickup_nb == nb_name, select = "dropoff_nb", drop = TRUE)

pickup_df <- rxDataStep(mht_xdf,
                        rowSelection = pickup_nb == nb & dropoff_nb %in% top_drop_for_nb,
                        varsToKeep = c("dropoff_nb", "pickup_datetime"),
                        transformObjects = list(nb = nb_name, top_drop_for_nb = nb_drop))

## ----chap06chunk19-------------------------------------------------------
library(scales)
library(lubridate)
pickup_df %>%
  mutate(pickup_hour = hour(ymd_hms(pickup_datetime, tz = "UTC"))) %>%
  ggplot(aes(x = pickup_hour, fill = dropoff_nb)) +
  geom_bar(position = "fill", stat = "count") +
  scale_fill_discrete(guide = guide_legend(reverse = TRUE)) +
  scale_y_continuous(labels = percent_format()) +
  ylab("percent")

## ----chap06chunk20-------------------------------------------------------
res1 <- rxCube(tip_percent ~ pickup_dow:pickup_hour, mht_xdf)
res2 <- rxCube(fare_amount/(trip_duration/60) ~ pickup_dow:pickup_hour, mht_xdf)
names(res2)[3] <- 'fare_per_minute'
res <- bind_cols(c(res1, res2))
res <- res[ , c('pickup_dow', 'pickup_hour', 'fare_per_minute', 'tip_percent', 'Counts')]

ggplot(res, aes(pickup_dow, pickup_hour)) +
  geom_tile(aes(fill = fare_per_minute), colour = "white") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(label = sprintf('%dK riders\n (%d%% tip)', 
                                signif(Counts/1000, 2), round(tip_percent, 0))), 
            size = 2.5) +
  coord_fixed(ratio = .9)

###################################################################################
# BUILDING MODELS
###################################################################################

## ----chap07chunk02-------------------------------------------------------
form_1 <- as.formula(tip_percent ~ pickup_nb:dropoff_nb + pickup_dow:pickup_hour)
rxlm_1 <- rxLinMod(form_1, data = mht_xdf, dropFirst = TRUE, covCoef = TRUE)

## ----chap07chunk03-------------------------------------------------------
rxs <- rxSummary( ~ pickup_nb + dropoff_nb + pickup_hour + pickup_dow, mht_xdf)
ll <- lapply(rxs$categorical, function(x) x[ , 1])
names(ll) <- c('pickup_nb', 'dropoff_nb', 'pickup_hour', 'pickup_dow')
pred_df_1 <- expand.grid(ll)
pred_df_1 <- rxPredict(rxlm_1, data = pred_df_1, computeStdErrors = TRUE, writeModelVars = TRUE)
names(pred_df_1)[1:2] <- paste(c('tip_pred', 'tip_stderr'), 1, sep = "_")
head(pred_df_1, 10)

## ----chap07chunk04-------------------------------------------------------
library(ggplot2)
ggplot(pred_df_1, aes(x = pickup_nb, y = dropoff_nb)) +
  geom_tile(aes(fill = tip_pred_1), colour = "white") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  scale_fill_gradient(low = "white", high = "red") +
  coord_fixed(ratio = .9)

## ----chap07chunk05-------------------------------------------------------
ggplot(pred_df_1, aes(x = pickup_dow, y = pickup_hour)) +
  geom_tile(aes(fill = tip_pred_1), colour = "white") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  scale_fill_gradient(low = "white", high = "red") +
  coord_fixed(ratio = .9)

## ----chap07chunk06-------------------------------------------------------
form_2 <- as.formula(tip_percent ~ pickup_nb:dropoff_nb)
rxlm_2 <- rxLinMod(form_2, data = mht_xdf, dropFirst = TRUE, covCoef = TRUE)
pred_df_2 <- rxPredict(rxlm_2, data = pred_df_1, computeStdErrors = TRUE, writeModelVars = TRUE)
names(pred_df_2)[1:2] <- paste(c('tip_pred', 'tip_stderr'), 2, sep = "_")

library(dplyr)
pred_df <- pred_df_2 %>%
  select(starts_with('tip_')) %>%
  cbind(pred_df_1) %>%
  arrange(pickup_nb, dropoff_nb, pickup_dow, pickup_hour) %>%
  select(pickup_dow, pickup_hour, pickup_nb, dropoff_nb, starts_with('tip_pred_'))

head(pred_df)

## ----chap07chunk07-------------------------------------------------------
ggplot(data = pred_df) +
  geom_density(aes(x = tip_pred_1, col = "complex")) +
  geom_density(aes(x = tip_pred_2, col = "simple")) +
  facet_grid(pickup_hour ~ pickup_dow)

## ----chap07chunk08-------------------------------------------------------
dfq <- data.frame(probs = seq(0, 1, by = .05))
dfq$tip_percent <- rxQuantile("tip_percent", data = mht_xdf, probs = dfq$probs)

ggplot(aes(x = tip_percent, y = probs), data = dfq) +
  geom_line()

## ----chap07chunk09-------------------------------------------------------
pred_df %>%
  mutate_each(funs(cut(., c(-Inf, 8, 12, 15, 18, Inf))), tip_pred_1, tip_pred_2) %>%
  ggplot() +
  geom_bar(aes(x = tip_pred_1, fill = "complex"), alpha = .5) +
  geom_bar(aes(x = tip_pred_2, fill = "simple"), alpha = .5) +
  facet_grid(pickup_hour ~ pickup_dow) +
  xlab('tip percent prediction') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

## ----chap07chunk10, eval=FALSE-------------------------------------------
## form_3 <- ## formula described above goes here
## rxlm_3 <- ## build a linear model based on the above formula

## ----chap07chunk11, eval=FALSE-------------------------------------------
## rxs <- rxSummary( ~ payment_type + pickup_nb + dropoff_nb + pickup_hour + pickup_dow, mht_xdf)
## ll <- lapply(rxs$categorical, function(x) x[ , 1])
## names(ll) <- c('payment_type', 'pickup_nb', 'dropoff_nb', 'pickup_hour', 'pickup_dow')
## pred_df <- expand.grid(ll)

## ----chap07chunk12, eval=FALSE-------------------------------------------
## ggplot(data = pred_all) +
##   geom_bar(aes(x = p1, fill = "model 1", group = payment_type), alpha = .5) +
##   geom_bar(aes(x = p3, fill = "model 3", group = payment_type), alpha = .5) +
##   facet_grid(pickup_hour ~ pickup_dow) +
##   xlab('tip percent prediction') +
##   theme(axis.text.x = element_text(angle = 90, hjust = 1))

## ----chap07chunk13-------------------------------------------------------
form_3 <- as.formula(tip_percent ~ payment_type + pickup_nb:dropoff_nb + pickup_dow:pickup_hour)
rxlm_3 <- rxLinMod(form_3, data = mht_xdf, dropFirst = TRUE, covCoef = TRUE)

## ----chap07chunk14-------------------------------------------------------
rxs <- rxSummary( ~ payment_type + pickup_nb + dropoff_nb + pickup_hour + pickup_dow, mht_xdf)
ll <- lapply(rxs$categorical, function(x) x[ , 1])
names(ll) <- c('payment_type', 'pickup_nb', 'dropoff_nb', 'pickup_hour', 'pickup_dow')
pred_df <- expand.grid(ll)

pred_df_1 <- rxPredict(rxlm_1, data = pred_df, computeStdErrors = TRUE, writeModelVars = TRUE)
pred_df_3 <- rxPredict(rxlm_3, data = pred_df, computeStdErrors = TRUE, writeModelVars = TRUE)

pred_df %>%
  cbind(select(rename(pred_df_1, p1 = tip_percent_Pred), p1)) %>%
  cbind(select(rename(pred_df_3, p3 = tip_percent_Pred), p3)) %>%
  mutate_at(vars(p1, p3), funs(cut(., c(-Inf, 8, 12, 15, 18, Inf)))) -> pred_all

## ----chap07chunk15-------------------------------------------------------
ggplot(data = pred_all) +
  geom_bar(aes(x = p1, fill = "model 1"), alpha = .5) +
  geom_bar(aes(x = p3, fill = "model 3"), alpha = .5) +
  facet_grid(~ payment_type) +
  xlab('tip percent prediction') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

## ----chap07chunk16-------------------------------------------------------
library(scales)
pred_df %>%
  cbind(select(rename(pred_df_1, p1 = tip_percent_Pred), p1)) %>%
  cbind(select(rename(pred_df_3, p3 = tip_percent_Pred), p3)) %>%
  mutate_at(vars(p1, p3), funs(rescale(., to = c(0, 20)))) %>%
  mutate_at(vars(p1, p3), funs(cut(., c(-Inf, 8, 12, 15, 18, Inf)))) -> pred_all

ggplot(data = pred_all) +
  geom_bar(aes(x = p1, fill = "model 1"), alpha = .5) +
  geom_bar(aes(x = p3, fill = "model 3"), alpha = .5) +
  facet_grid(~ payment_type) +
  xlab('tip percent prediction') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

## ----chap07chunk17-------------------------------------------------------
dir.create(file.path(data_dir, 'output'), showWarnings = FALSE)
rx_split_xdf <- function(xdf = mht_xdf, split_perc = 0.75,
                         output_path = file.path(data_dir, "output"), ...) {
  # first create a column to split by
  outFile <- tempfile(fileext = 'xdf')
  rxDataStep(inData = xdf, outFile = xdf, transforms = list(
    split = factor(ifelse(rbinom(.rxNumRows, size = 1, prob = splitperc), "train", "test"))),
    transformObjects = list(splitperc = split_perc), overwrite = TRUE, ...)
  
  # then split the data in two based on the column we just created
  splitDS <- rxSplit(inData = xdf,
                     outFilesBase = file.path(output_path, "train"),
                     splitByFactor = "split",
                     overwrite = TRUE)
  return(splitDS)
}

# we can now split to data in two
mht_split <- rx_split_xdf(xdf = mht_xdf, 
                          varsToKeep = c('payment_type', 'fare_amount', 'tip_amount', 'tip_percent', 
                                         'pickup_hour', 'pickup_dow', 'pickup_nb', 'dropoff_nb'))
names(mht_split) <- c("train", "test")

## ----chap07chunk18-------------------------------------------------------
system.time(linmod <- rxLinMod(tip_percent ~ pickup_nb:dropoff_nb + pickup_dow:pickup_hour, data = mht_split$train, reportProgress = 0))
system.time(dtree <- rxDTree(tip_percent ~ pickup_nb + dropoff_nb + pickup_dow + pickup_hour, data = mht_split$train, pruneCp = "auto", reportProgress = 0))
system.time(dforest <- rxDForest(tip_percent ~ pickup_nb + dropoff_nb + pickup_dow + pickup_hour, data = mht_split$train, nTree = 30, importance = TRUE, useSparseCube = TRUE, reportProgress = 0))

## ----chap07chunk19-------------------------------------------------------
trained.models <- list(linmod = linmod, dtree = dtree, dforest = dforest)
save(trained.models, file = 'trained_models.Rdata')

## ----chap07chunk20-------------------------------------------------------
pred_df <- expand.grid(ll[2:5])
pred_df_1 <- rxPredict(trained.models$linmod, data = pred_df, predVarNames = "pred_linmod")
pred_df_2 <- rxPredict(trained.models$dtree, data = pred_df, predVarNames = "pred_dtree")
pred_df_3 <- rxPredict(trained.models$dforest, data = pred_df, predVarNames = "pred_dforest")
pred_df <- do.call(cbind, list(pred_df, pred_df_1, pred_df_2, pred_df_3))

observed_df <- rxSummary(tip_percent ~ pickup_nb:dropoff_nb:pickup_dow:pickup_hour, mht_xdf)
observed_df <- observed_df$categorical[[1]][ , c(2:6)]
pred_df <- inner_join(pred_df, observed_df, by = names(pred_df)[1:4])

ggplot(data = pred_df) +
  geom_density(aes(x = Means, col = "observed average"), size = 1) +
  geom_density(aes(x = pred_linmod, col = "linmod"), size = 1) +
  geom_density(aes(x = pred_dtree, col = "dtree"), size = 1) +
  geom_density(aes(x = pred_dforest, col = "dforest"), size = 1) +
  xlim(-1, 30) +
  xlab("tip percent")

## ----chap07chunk21-------------------------------------------------------
rxPredict(trained.models$linmod, data = mht_split$test, outData = mht_split$test, predVarNames = "tip_percent_pred_linmod", overwrite = TRUE)
rxPredict(trained.models$dtree, data = mht_split$test, outData = mht_split$test, predVarNames = "tip_percent_pred_dtree", overwrite = TRUE)
rxPredict(trained.models$dforest, data = mht_split$test, outData = mht_split$test, predVarNames = "tip_percent_pred_dforest", overwrite = TRUE)

rxSummary(~ SSE_linmod + SSE_dtree + SSE_dforest, data = mht_split$test,
          transforms = list(
            SSE_linmod = (tip_percent - tip_percent_pred_linmod)^2,
            SSE_dtree = (tip_percent - tip_percent_pred_dtree)^2,
            SSE_dforest = (tip_percent - tip_percent_pred_dforest)^2))

## ----chap07chunk22-------------------------------------------------------
rxc <- rxCor( ~ tip_percent + tip_percent_pred_linmod + tip_percent_pred_dtree + tip_percent_pred_dforest, data = mht_split$test)
print(rxc)

## ----chap07chunk23, eval=FALSE-------------------------------------------
## # we re-build the linear model without payment_type for comparison
## linmod_1 <- rxLinMod(## formula goes here
##                      data = mht_split$train, reportProgress = 0)
## # we build a linear model with payment_type
## linmod_2 <- rxLinMod(## formula goes here
##                      data = mht_split$train, reportProgress = 0)
## # we build a decision tree with payment_type
## dtree <- rxDTree(## formula goes here
##                  data = mht_split$train, pruneCp = "auto", reportProgress = 0)
## 
## trained.models <- list(linmod_1 = linmod_1, linmod_2 = linmod_2, dtree = dtree)

## ----chap07chunk24, eval=FALSE-------------------------------------------
## pred_df <- expand.grid(ll)
## ## predict payment_type on pred_df by each of three models
## # pred_df_1 stores predictions made by linmod_1 into a column called pred_linmod_1
## # pred_df_2 stores predictions made by linmod_2 into a column called pred_linmod_2
## # pred_df_3 stores predictions made by dtree into a column called pred_dtree
## pred_df <- bind_cols(pred_df, pred_df_1, pred_df_2, pred_df_3)
## head(pred_df)

## ----chap07chunk25, eval=FALSE-------------------------------------------
## ggplot(data = pred_df) +
##   geom_density(aes(x = pred_linmod_1, col = "linmod_1"), size = 1) +
##   geom_density(aes(x = pred_linmod_2, col = "linmod_2"), size = 1) +
##   geom_density(aes(x = pred_dtree, col = "dtree"), size = 1) +
##   xlim(-10, 40) +
##   xlab("tip percent") +
##   facet_grid(payment_type ~ ., scales = "free")

## ----chap07chunk26, eval=FALSE-------------------------------------------
## test_df <- rxDataStep(mht_split$test,
##   varsToKeep = c('tip_percent', 'payment_type', 'pickup_nb', 'dropoff_nb', 'pickup_hour', 'pickup_dow'),
##   maxRowsByCols = 10^9)
## test_df_1 <- ## predict for the first model, call the predictions `tip_pred_linmod`
## test_df_2 <- ## predict for the second model, call the predictions `tip_pred_dtree`
## test_df_3 <- ## predict for the third model, call the predictions `tip_pred_dforest`

## ----chap07chunk27, eval=FALSE-------------------------------------------
## test_df <- do.call(cbind, list(test_df, test_df_1, test_df_2, test_df_3))
## 
## rxSummary(~ SSE_linmod + SSE_dtree + SSE_dforest, data = test_df,
##   transforms = list(
##     SSE_linmod = (tip_percent - tip_pred_linmod)^2,
##     SSE_dtree = (tip_percent - tip_pred_dtree)^2,
##     SSE_dforest = (tip_percent - tip_pred_dforest)^2))

## ----chap07chunk28-------------------------------------------------------
linmod_1 <- rxLinMod(tip_percent ~ pickup_nb:dropoff_nb + pickup_dow:pickup_hour, data = mht_split$train, reportProgress = 0)
linmod_2 <- rxLinMod(tip_percent ~ payment_type + pickup_nb:dropoff_nb + pickup_dow:pickup_hour, data = mht_split$train, reportProgress = 0)
dtree <- rxDTree(tip_percent ~ payment_type + pickup_nb + dropoff_nb + pickup_dow + pickup_hour, data = mht_split$train, pruneCp = "auto", reportProgress = 0)

trained.models <- list(linmod_1 = linmod_1, linmod_2 = linmod_2, dtree = dtree)

## ----chap07chunk29-------------------------------------------------------
pred_df <- expand.grid(ll)
pred_df_1 <- rxPredict(trained.models$linmod_1, data = pred_df, predVarNames = "pred_linmod_1")
pred_df_2 <- rxPredict(trained.models$linmod_2, data = pred_df, predVarNames = "pred_linmod_2")
pred_df_3 <- rxPredict(trained.models$dtree, data = pred_df, predVarNames = "pred_dtree")
pred_df <- bind_cols(pred_df, pred_df_1, pred_df_2, pred_df_3)
head(pred_df)

## ----chap07chunk30-------------------------------------------------------
ggplot(data = pred_df) +
  geom_density(aes(x = pred_linmod_1, col = "linmod_1"), size = 1) +
  geom_density(aes(x = pred_linmod_2, col = "linmod_2"), size = 1) +
  geom_density(aes(x = pred_dtree, col = "dtree"), size = 1) +
  xlim(-10, 40) +
  xlab("tip percent") +
  facet_grid(payment_type ~ ., scales = "free")

## ----chap07chunk31-------------------------------------------------------
test_df <- rxDataStep(mht_split$test, varsToKeep = c('tip_percent', 'payment_type', 'pickup_nb', 'dropoff_nb', 'pickup_hour', 'pickup_dow'), maxRowsByCols = 10^9)
test_df_1 <- rxPredict(trained.models$linmod_1, data = test_df, predVarNames = "tip_pred_linmod_1")
test_df_2 <- rxPredict(trained.models$linmod_2, data = test_df, predVarNames = "tip_pred_linmod_2")
test_df_3 <- rxPredict(trained.models$dtree, data = test_df, predVarNames = "tip_pred_dtree")
test_df <- do.call(cbind, list(test_df, test_df_1, test_df_2, test_df_3))
head(test_df)

## ----chap07chunk32-------------------------------------------------------
ggplot(data = test_df) +
  geom_density(aes(x = tip_pred_linmod_1, col = "linmod w/o payment type")) +
  geom_density(aes(x = tip_pred_linmod_2, col = "linmod w/ payment type")) +
  geom_density(aes(x = tip_pred_dtree, col = "dtree with payment type")) +
  xlab("tip percent") # + facet_grid(pickup_hour ~ pickup_dow)

## ----chap07chunk33-------------------------------------------------------
rxSummary(~ SSE_linmod_1 + SSE_linmod_2 + SSE_dtree, data = test_df,
          transforms = list(
            SSE_linmod_1 = (tip_percent - tip_pred_linmod_1)^2,
            SSE_linmod_2 = (tip_percent - tip_pred_linmod_2)^2,
            SSE_dtree = (tip_percent - tip_pred_dtree)^2))

## ----chap07chunk34-------------------------------------------------------
rxc <- rxCor( ~ tip_percent + tip_pred_linmod_1 + tip_pred_linmod_2 + tip_pred_dtree, data = test_df)
print(rxc)


