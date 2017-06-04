##############
#4-1-1.R
##############
library(dmr.stats)
library(dmr.util)

data(weather, package="dmr.data")
data(weatherc, package="dmr.data")
##############
#4-2-1.R
##############
## calculate the posterior probability given prior and inverse probabilities
bayes.rule <- function(prior, inv)
{
  prior*inv/sum(prior*inv)
}

# posterior probabilities
bayes.rule(c(0.2, 0.3, 0.5), c(0.9, 0.9, 0.5))

# let P(burglery)=0.001,
# P(alarm|burglery)=0.95,
# P(alarm|not burglery)=0.005
# calculate P(burglery|alarm)
bayes.rule(c(0.001, 0.999), c(0.95, 0.005))[1]
##############
#4-3-1.R
##############
# prior class probabilities for the weather data
pdisc(weather$play)
##############
#4-3-2.R
##############
# conditional attribute value probabilities given the class
pcond(weather$outlook, weather$play)
pcond(weather$temperature, weather$play)
pcond(weather$humidity, weather$play)
pcond(weather$wind, weather$play)
##############
#4-3-3.R
##############
## create a naive Bayes classifier
nbc <- function(formula, data)
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  
  cc <- integer(nlevels(data[[class]]))  # initialize class counts
  names(cc) <- levels(data[[class]])
  avc <- sapply(attributes,              # initialize attribute-value-class counts
                function(a)
                  matrix(0, nrow=nlevels(data[[a]]), ncol=nlevels(data[[class]]),
                         dimnames=list(levels(data[[a]]), levels(data[[class]]))))
  
  for (i in (1:nrow(data)))              # iterate through training instances
  {
    cc[data[[class]][i]] <- cc[data[[class]][i]]+1  # increment class count
    for (a in attributes)                # increment attribute-value-class counts
      avc[[a]][data[[a]][i],data[[class]][i]] <-
        avc[[a]][data[[a]][i],data[[class]][i]]+1
  }
  
  # calculate probability estimates based on counts
  `class<-`(list(prior=cc/sum(cc),
                 cond=sapply(avc, function(avc1)
                   t(apply(avc1, 1, "/", colSums(avc1))))),
            "nbc")
}

# naive Bayes classifier for the weather data
nbw <- nbc(play~., weather)
##############
#4-3-4.R
##############
## naive Bayes prediction for a single instance
predict1.nbc <- function(model, x)
{
  aind <- names(x) %in% names(model$cond)
  bnum <- model$prior*apply(mapply(function(a, v)
    model$cond[[a]][v,], names(model$cond), x[aind]),
    1, prod)
  bnum/sum(bnum)
}

## naive Bayes prediction for a dataset
predict.nbc <- function(model, data)
{
  t(sapply(1:nrow(data), function(i) predict1.nbc(model, data[i,])))
}

# make predictions for the weather data
predict(nbw, weather)
##############
#4-4-1.R
##############
## m-estimated conditional probability distribution P(v1|v2)
mpcond <- function(v1, v2, p=1/nlevels(v1), m=nlevels(v1))
{
  count <- table(v1, v2, dnn=NULL)
  t(apply(count, 1, function(cnt, sumcnt) mest(cnt, sumcnt, p, m), colSums(count)))
}

# conditional attribute value probabilities given the class
pcond(weather$outlook, weather$play)
mpcond(weather$outlook, weather$play)
mpcond(weather$outlook, weather$play, m=0)
mpcond(weather$outlook, weather$play, m=1)
##############
#4-4-2.R
##############
## conditional mean
mcond <- function(v1, v2)
{
  tapply(v1, v2, mean)
}

## conditional variance
vcond <- function(v1, v2)
{
  tapply(v1, v2, var)
}

# conditional mean and variance of attribute values given the class
mcond(weatherc$temperature, weatherc$play)
vcond(weatherc$temperature, weatherc$play)
mcond(weatherc$humidity, weatherc$play)
vcond(weatherc$humidity, weatherc$play)
##############
#4-4-3.R
##############
# weather data with missing values
weatherm <- weather
weatherm$outlook[1] <- NA
weatherm$humidity[1:2] <- NA

# conditional attribute value probabilities given the class
# with and without missing values
pcond(weather$outlook, weather$play)
pcond(weatherm$outlook, weatherm$play)
pcond(weather$humidity, weather$play)
pcond(weatherm$humidity, weatherm$play)