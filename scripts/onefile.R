pkglist<-c('devtools','remotes','e1071','rpart','rpart.plot','lattice','ipred','cluster','quadprog',
           'kernlab','Matrix','randomForest','nnet')
dmrpkglist<-c('dmr.data','dmr.util','dmr.util','dmr.claseval','dmr.stats',
              'dmr.trans','dmr.linreg','dmr.regeval','dmr.dissim','dmr.dectree',
              'dmr.linclas','dmr.disc','dmr.kcenters','dmr.cluseval','dmr.regtree',
              'dmr.attrsel','dmr.ensemble','dmr.kernel','dmr.bayes','dmr.hierclus',
              'dmr.miscost','dmr.rpartutil')
pkgcheck <- pkglist %in% row.names(installed.packages())
library(devtools)
#install.packages(pkglist[!pkgcheck],depend=TRUE) 
#install_github(paste("42n4/", dmrpkglist, sep=""))

for(i in pkglist){ library(i, character.only = TRUE);}
for(i in dmrpkglist){ library(i, character.only = TRUE);}
data(weather, package="dmr.data")
data(weatherc, package="dmr.data")
data(weatherr, package="dmr.data")

#packages for my tutorial https://github.com/pwasiewi/earin/blob/master/scripts/01stepfirsten.R
pkglist<-c(pkglist,"reshape","ade4","sqldf","plyr","dplyr")
pkglist<-c(pkglist,"party","rgl","scatterplot3d","fpc","pvclust","dendextend")
pkglist<-c(pkglist,"nFactors","FactoMineR","RRF","mclust","foreach","doParallel")
pkglist<-c(pkglist,"rpart","ipred","gbm","mda","klaR","kernlab","caret")
pkglist<-c(pkglist,"tseries","fUnitRoots","forecast","sets","TTR")
#pkglist<-c(pkglist,"MASS","RWeka")
pkgcheck <- pkglist %in% row.names(installed.packages())
#packages still not installed
paste(pkglist[!pkgcheck],collapse=' ')
for(i in pkglist){ library(i, character.only = TRUE);}
##############
#2-4-10.R
##############
weighted.var <- function(v, w=rep(1, length(v)))
{
  sw <- sum(w)
  ssw <- sum(w^2)
  wm <- weighted.mean(v, w)
  sw/(sw^2-ssw)*sum(w*(v-wm)^2)
}

  # demonstration
weighted.var(weatherr$playability)
weighted.var(weatherr$playability, ifelse(weatherr$outlook=="rainy", 2, 1))
##############
#2-4-11.R
##############
## weighted variance that returns 0 for 1-element vectors and NaN for empty vectors
weighted.var1 <- function(v, w=rep(1, length(w)))
{ switch(min(length(v), 2)+1, NaN, 0, weighted.var(v, w)) }

  # demonstration
weighted.var1(1:2, 1:2)
weighted.var1(1, 2)
weighted.var1(weatherr$temperature[weatherr$playability<0.75],
              weatherr$playability[weatherr$playability<0.75])
weighted.var1(weatherr$temperature[weatherr$playability>=0.75],
              weatherr$playability[weatherr$playability>=0.75])
weighted.var1(weatherr$temperature[weatherr$playability>=0.8],
              weatherr$playability[weatherr$playability>=0.8])
##############
#2-4-12.R
##############
bs.sd <- function(v) { sqrt(sum((v-mean(v))^2)/(length(v)-1)) }

  # demonstration
bs.sd(weatherr$playability)
sd(weatherr$playability)
##############
#2-4-13.R
##############
varcoef <- function(v) { sqrt(sum((v-(m <- mean(v)))^2)/(length(v)-1))/m }

  # demonstration
varcoef(weatherr$playability)
varcoef(-weatherr$playability)
##############
#2-4-14.R
##############
relsd <- function(v) { abs(varcoef(v)) }

  # demonstration
relsd(weatherr$playability)
relsd(-weatherr$playability)
##############
#2-4-15.R
##############
bs.mad <- function(v, scale=1/qnorm(0.75)) { scale*median(abs(v-median(v))) }

  # demonstration
bs.mad(weatherr$playability, scale=1)
mad(weatherr$playability, constant=1)
bs.mad(weatherr$playability)
mad(weatherr$playability)
sd(weatherr$playability)
##############
#2-4-16.R
##############
iqr <- function(v)  { unname(diff(quantile(v, c(0.25, 0.75)))) }

  # demonstration
iqr(weatherc$temperature)
##############
#2-4-17.R
##############
qd <- function(v) { unname(diff(q <- quantile(v, c(0.25, 0.75)))/sum(q)) }

  # demonstration
qd(weatherc$temperature)
##############
#2-4-18.R
##############
is.outlier <- function(v, b=1.5)
{ v<(q <- quantile(v, c(0.25, 0.75)))[1]-b*(r <- diff(q)) | v>q[2]+b*r }

weatherc$temperature[is.outlier(weatherc$temperature, 0.5)]
boxplot(weatherc$temperature, range=0.5, plot=FALSE)
boxplot(weatherc$temperature, range=0.49, plot=FALSE)
##############
#2-4-19.R
##############
modal <- function(v)
{
  m <- which.max(table(v))
  if (is.factor(v))
    flevels(v)[m]
  else
    sort(unique(v))[m]
}

  # demonstration
modal(weather$outlook)
modal(weatherr$temperature)
##############
#2-4-1.R
##############
bs.mean <- function(v) { sum(v)/length(v) }

  # demonstration
bs.mean(weatherc$temperature)
mean(weatherc$temperature)
##############
#2-4-20.R
##############
weighted.modal <- function(v, w=rep(1, length(v)))
{
  m <- which.max(weighted.table(v, w=w))
  if (is.factor(v))
    factor(levels(v)[m], levels=levels(v))
  else
    sort(unique(v))[m]
}

  # demonstration
weighted.modal(weather$outlook)
weighted.modal(weather$outlook, w=ifelse(weather$play=="yes", 2, 1))
##############
#2-4-21.R
##############
prob <- function(v, v1) { sum(v==v1)/length(v) }

  # demonstration
prob(weather$outlook, "rainy")
##############
#2-4-22.R
##############
pdisc <- function(v, ...) { (count <- table(v, ..., dnn=NULL))/sum(count) }

  # demonstration
pdisc(weather$outlook)
pdisc(weather$outlook, weather$temperature)
##############
#2-4-23.R
##############
## conditional probability distribution P(v1|v2)
pcond <- function(v1, v2)
{
  t(apply(count <- table(v1, v2, dnn=NULL), 1, "/", colSums(count)))
}

  # demonstration
pcond(weather$outlook, weather$play)
##############
#2-4-24.R
##############
weighted.prob <- function(v, v1, w=rep(1, length(v))) { sum(w[v==v1])/sum(w) }

  # demonstration
weighted.prob(weather$outlook, "rainy")
weighted.prob(weather$outlook, "rainy", w=ifelse(weather$play=="yes", 2, 1))
##############
#2-4-25.R
##############
## weighted discrete probability distribution
weighted.pdisc <- function(v, ..., w=rep(1:length(v)))
{
  (count <- weighted.table(v, ..., w=w))/sum(count)
}

  # demonstration
weighted.pdisc(weather$outlook, w=ifelse(weather$play=="yes", 2, 1))
weighted.pdisc(weather$outlook, weather$temperature,
               w=ifelse(weather$play=="yes", 2, 1))
##############
#2-4-26.R
##############
## entropy for discrete probability distributions
entropy.p <- function(p) { sum(-plogp(p)) }

entropy <- function(v) { entropy.p(pdisc(v)) }

  # demonstration
entropy.p(c(1/5, 2/5, 3/5))
entropy(weather$outlook)
entropy(weather$play)
entropy(weather$play[weather$outlook=="overcast"])
entropy(weather$play[weather$outlook!="overcast"])
##############
#2-4-27.R
##############
## Gini index for discrete probability distributions
gini.p <- function(p) { 1-sum(p^2) }

gini <- function(v) { gini.p(pdisc(v)) }

  # demonstration
gini.p(c(1/5, 2/5, 3/5))
gini(weather$outlook)
gini(weather$play)
gini(weather$play[weather$outlook=="overcast"])
gini(weather$play[weather$outlook!="overcast"])
##############
#2-4-28.R
##############
  # plot the entropy
curve(-x*log(x)-(1-x)*log(1-x), from=0, to=1,
      xlab="p", ylab="", ylim=c(-0.02, 0.7), lty=1)
  # and add the plot of the Gini index
curve(1-x^2-(1-x)^2, from=0, to=1, add=TRUE, lty=2)
legend("topright", legend=c("entropy", "gini"), lty=1:2)
##############
#2-4-29.R
##############
prob.ci.par <- function(v, v1=1, delta=0.05)
{
  list(p=(p <- prob(v, v1)),
       low=p-(u <- qnorm(1-delta/2))*(sp <- sqrt(p*(1-p)/length(v))),
       high=p+u*sp)
}

  # demonstration
prob.ci.par(weather$play, "yes")
prob.ci.par(weather$play, "yes", delta=0.01)
prob.ci.par(weather$play, "yes", delta=0.1)
##############
#2-4-2.R
##############
bs.weighted.mean <- function(v, w=rep(1, length(v))) { sum(w*v)/sum(w) }

  # demonstration
bs.weighted.mean(weatherc$temperature, ifelse(weatherc$play=="yes", 5, 1))
weighted.mean(weatherc$temperature, ifelse(weatherc$play=="yes", 5, 1))
##############
#2-4-30.R
##############
prob.ci.boot <- function(v, v1=1, delta=0.05, m=1000)
{
  q <- unname(quantile(sapply(1:m, function(i) prob(sample(v, replace=TRUE), v1)),
                       probs=c(delta/2, 1-delta/2)))
  list(p=prob(v, v1), low=q[1], high=q[2])
}

  # demonstration
prob.ci.boot(weather$play, "yes")
prob.ci.boot(weather$play, "yes", delta=0.01)
prob.ci.boot(weather$play, "yes", delta=0.1)
##############
#2-4-31.R
##############
## m-estimate of probability of an event occurring n1 out of n times
## incorporating m fictitious instances
mest <- function(n1, n, m=2, p0=1/m) { (n1+m*p0)/(n+m) }

mprob <- function(v, v1, m=nlevels(v), p0=1/nlevels(v))
{ mest(sum(v==v1), length(v), m, p0) }

  # demonstration
mest(0, 10, 1, 0.5)
mest(0, 10, 2, 0.5)
mest(10, 10, 1, 0.5)
mest(10, 10, 2, 0.5)

mprob(weather$outlook, "rainy", m=0)
mprob(weather$outlook, "rainy")
mprob(weather$play[weather$outlook=="overcast"], "no", m=0)
mprob(weather$play[weather$outlook=="overcast"], "no")
mprob(weather$play[weather$outlook=="overcast"], "no", m=3, p0=0.5)
##############
#2-4-32.R
##############
## Laplace estimate of probability of an event occurring n1 out of n times
## with m possible outcomes
laest <- function(n1, n, m=2) { mest(n1, n, m)  }

laprob <- function(v, v1) { mprob(v, v1) }

  # demonstration
laest(0, 10, 2)
mest(0, 10, 2)
laest(10, 10, 2)
mest(10, 10, 2)

laprob(weather$outlook, "rainy")
mprob(weather$outlook, "rainy", m=3, p0=1/3)
laprob(weather$play[weather$outlook=="overcast"], "no")
mprob(weather$play[weather$outlook=="overcast"], "no", m=2, p0=0.5)
##############
#2-4-33.R
##############
## m-mean that incorporates m fictitious values with a specified mean m0
mmean <- function(v, m=2, m0=mean(v)) { (sum(v)+m*m0)/(length(v)+m) }

  # demonstration
mmean(weatherr$playability)
mmean(weatherr$playability, m=0)
mmean(weatherr$playability, m0=0.5)
mmean(weatherr$playability, 5, 0.5)
mmean(weatherr$playability[weatherr$temperature<25], m=0)
mmean(weatherr$playability[weatherr$temperature<25], m0=mean(weatherr$playability))
##############
#2-4-34.R
##############
## m-variance that incorporates m fictitious values with a specified variance s02
mvar <- function(v, m=2, m0=mean(v), s02=var(v))
{ (sum((v-mmean(v, m, m0))^2)+max(m-1, 0)*s02)/max(length(v)+m-2, 1)  }

  # demonstration
mvar(weatherr$playability)
mvar(weatherr$playability, m=0)
mvar(weatherr$playability, s02=0.05)
mvar(weatherr$playability, m=5, s02=0.05)
mvar(weatherr$playability[weatherr$temperature<25], m=0)
mvar(weatherr$playability[weatherr$temperature<25],
     m0=mean(weatherr$playability), s02=var(weatherr$playability))
##############
#2-4-3.R
##############
bs.median <- function(v)
{
  k1 <- (m <- length(v))%/%2+1
  k2 <- (m+1)%/%2
  ((v <- sort(v))[k1]+v[k2])/2
}

  # demonstration
bs.median(weatherc$temperature)
bs.median(weatherc$temperature[weatherc$play=="yes"])
median(weatherc$temperature)
median(weatherc$temperature[weatherc$play=="yes"])
##############
#2-4-4.R
##############
weighted.median <- function(v, w=rep(1, length(v)))
{
  v <- v[ord <- order(v)]
  w <- w[ord]
  tw <- (sw <- cumsum(w))[length(sw)]
  mean(v[which(sw>=0.5*tw & tw-shift.right(sw, 0)>=0.5*tw)])
}

  # demonstration
weighted.median(weatherc$temperature, ifelse(weatherc$play=="yes", 5, 1))
median(c(weatherc$temperature[weatherc$play=="no"],
         rep(weatherc$temperature[weatherc$play=="yes"], 5)))
weighted.median(weatherc$temperature, ifelse(weatherc$play=="yes", 0.2, 1))
median(c(weatherc$temperature[weatherc$play=="yes"],
         rep(weatherc$temperature[weatherc$play=="no"], 5)))
##############
#2-4-5.R
##############
bs.rank <- function(v)
{
  r.min <- match(v, sort(v))
  r.max <- length(v)+1-match(v, rev(sort(v)))
  (r.min+r.max)/2
}

  # demonstration
bs.rank(weatherr$playability)
rank(weatherr$playability)
##############
#2-4-6.R
##############
ord <- function(v, k=1:length(v))
{
  sort(v)[k]
}

  # demonstration
ord(weatherr$playability, 11)
weatherr$playability[rank(weatherr$playability, ties.method="first")==11]
ord(weatherr$playability, 10:13)
weatherr$playability[rank(weatherr$playability, ties.method="first") %in% 10:13]
##############
#2-4-7.R
##############
bs.quantile <- function(v, p=c(0, 0.25, 0.5, 0.75, 1))
{
  b <- 1-p
  k <- floor((ps <- p*length(v))+b)
  beta <- ps+b-k
  `names<-`((1-beta)*(v <- sort(v))[k]+beta*(ifelse(k<length(v), v[k+1], v[k])), p)
}

  # demonstration
bs.quantile(weatherc$temperature)
quantile(weatherc$temperature)
bs.quantile(weatherc$temperature[weatherc$play=="yes"])
quantile(weatherc$temperature[weatherc$play=="yes"])
##############
#2-4-8.R
##############
bs.var <- function(v) { sum((v-mean(v))^2)/(length(v)-1) }

  # demonstration
bs.var(weatherr$playability)
var(weatherr$playability)
##############
#2-4-9.R
##############
## variance that returns 0 for 1-element vectors and NaN for empty vectors
var1 <- function(v) { switch(min(length(v), 2)+1, NaN, 0, var(v)) }

  # demonstration
var1(1:2)
var1(1)
var1(weatherr$temperature[weatherr$playability<0.75])
var1(weatherr$temperature[weatherr$playability>=0.75])
var1(weatherr$temperature[weatherr$playability>=0.8])
##############
#2-5-10.R
##############
bs.wilcox.test <- function(v, v01)
{
  subsets <- split(v, v01)
  ranks <- unname(split(rank(v), v01))
  cn <- unname(sapply(subsets, length))
  mu <- cn[1]*cn[2]/2
  su <- sqrt(cn[1]*cn[2]*(cn[1]+cn[2]+1)/12)

  u <- sum(ranks[[1]])-cn[1]*(cn[1]+1)/2
#  u <- sum(sapply(subsets[[2]],
#                  function(v2) sum(v2<subsets[[1]])+sum(v2==subsets[[1]])/2))
  list(statistic=u, p.value=2*(1-pnorm(abs(u-mu)/su)))
}

  # demonstration
bs.wilcox.test(weatherc$temperature, weatherc$play)
wilcox.test(temperature~play, weatherc, exact=FALSE, correct=FALSE)
##############
#2-5-11.R
##############
bs.kruskal.test <- function(v1, v2)
{
  subsets <- split(v1, v2)
  ranks <- split((rank.all <- rank(v1)), v2)
  cn <- unname(sapply(subsets, length))
  mr <- sapply(ranks, mean)
  mr.a <- mean(rank.all)

  k <- (length(v1)-1)*sum(cn*(mr-mr.a)^2)/
         sum(sapply(ranks, function(r) sum((r-mr.a)^2)))
  list(statistic=k, p.value=1-pchisq(k, length(subsets)-1))
}

  # demonstration
bs.kruskal.test(weatherc$temperature, weatherc$play)
kruskal.test(temperature~play, weatherc)
bs.kruskal.test(weatherc$temperature, weatherc$outlook)
kruskal.test(temperature~outlook, weatherc)
##############
#2-5-1.R
##############
corl.test <- function(v1, v2)
{
  rho <- sum((v1-(m1 <- mean(v1)))*(v2-(m2 <- mean(v2))))/
           sqrt(sum((v1-m1)^2)*sum((v2-m2)^2))
  ts <- rho*sqrt((df <- length(v1)-2)/(1-rho^2))
  list(rho=rho, statistic=ts, p.value=2*(1-pt(abs(ts), df)))
}

  # demonstration
corl.test(weatherr$temperature, weatherr$playability)
cor.test(weatherr$temperature, weatherr$playability, method="pearson")
corl.test(weatherr$temperature, -weatherr$playability)
cor.test(weatherr$temperature, -weatherr$playability, method="pearson")
##############
#2-5-2.R
##############
corr.test <- function(v1, v2) { corl.test(rank(v1), rank(v2)) }

  # demonstration
corr.test(weatherr$temperature, weatherr$playability)
cor.test(weatherr$temperature, weatherr$playability, method="spearman")
corr.test(weatherr$temperature, -weatherr$playability)
cor.test(weatherr$temperature, -weatherr$playability, method="spearman")
##############
#2-5-3.R
##############
bs.chisq.test <- function(v1, v2)
{
  o12 <- table(v1, v2)
  e12 <- table(v1)%*%t(table(v2))/sum(o12)
  chi2 <- sum((o12-e12)^2/e12)
  list(statistic=chi2, p.value=1-pchisq(chi2, (nrow(o12)-1)*(ncol(o12)-1)))
}

  # demonstration
bs.chisq.test(weather$outlook, weather$play)
chisq.test(weather$outlook, weather$play)
##############
#2-5-4.R
##############
g.test <- function(v1, v2)
{
  o12 <- table(v1, v2)
  e12 <- table(v1)%*%t(table(v2))/sum(o12)
  g <- 2*sum(o12*log(o12/e12), na.rm=TRUE)
  list(statistic=g, p.value=1-pchisq(g, (nrow(o12)-1)*(ncol(o12)-1)))
}

  # demonstration
g.test(weather$outlook, weather$play)
##############
#2-5-5.R
##############
entropy.cond <- function(v1, v2)
{
  p12 <- pdisc(v1, v2)
  p2 <- colSums(p12)
  sum(p2*mapply(function(i, p2i) entropy.p(p12[,i]/p2i), 1:ncol(p12), p2))
}

  # demonstration
entropy.cond(weather$play, weather$outlook)
entropy.cond(weather$play, weather$outlook=="rainy")
##############
#2-5-6.R
##############
mutinfo <- function(v1, v2)
{
  p12 <- pdisc(v1, v2)
  p1 <- rowSums(p12)
  p2 <- colSums(p12)
  sum(p12*logp(p12/(p1%o%p2)), na.rm=TRUE)
}

  # demonstration
mutinfo(weather$outlook, weather$play)
  # this should be the same
entropy(weather$play)-entropy.cond(weather$play, weather$outlook)
entropy(weather$outlook)-entropy.cond(weather$outlook, weather$play)
g.test(weather$outlook, weather$play)$statistic/(2*log(2)*nrow(weather))
##############
#2-5-7.R
##############
## symmetric uncertainty for discrete vectors
symunc <- function(v1, v2)
{
  2*mutinfo(v1, v2)/(entropy(v1)+entropy(v2))
}

  # demonstration
symunc(weather$outlook, weather$temperature)
symunc(weather$outlook, weather$play)
##############
#2-5-8.R
##############
bs.t.test <- function(v, v01)
{
  m <- unname(tapply(v, v01, mean))
  s2 <- unname(tapply(v, v01, var))
  cn <- unname(tapply(v, v01, length))
  sp <- sqrt((s2[1]*(cn[1]-1)+s2[2]*(cn[2]-1))/(sum(cn)-2))

  ts <- (m[1]-m[2])/(sp*sqrt(sum(1/cn)))
  list(statistic=ts, p.value=2*(1-pt(abs(ts), sum(cn)-2)))
}

  # demonstration
bs.t.test(weatherc$temperature, weatherc$play)
t.test(temperature~play, weatherc, var.equal=TRUE)
##############
#2-5-9.R
##############
f.test <- function(v1, v2)
{
  subsets <- split(v1, v2)
  m <- unname(sapply(subsets, mean))
  cn <- unname(sapply(subsets, length))
  m.a <- mean(v1)
  cn.a <- length(v1)

  f <- (sum(cn*(m-m.a)^2)/((k <- length(subsets))-1))/
         (sum(sapply(1:length(subsets),
                     function(i) sum((subsets[[i]]-m[i])^2)))/((cn.a-k)))
  list(statistic=f, p.value=1-pf(f, k-1, cn.a-k))
}

  # demonstration
f.test(weatherc$temperature, weatherc$outlook)
f.test(weatherc$temperature, weatherc$play)
anova(lm(temperature~outlook, weatherc))
anova(lm(temperature~play, weatherc))
abs(sqrt(f.test(weatherc$temperature, weatherc$play)$statistic)-
  abs(t.test(temperature~play, weatherc, var.equal=TRUE)$statistic))
##############
#2-6-1.R
##############
par(mfrow=c(1, 2))
boxplot(weatherr$playability, range=0.5, col="grey", main="playability")
boxplot(playability~outlook, weatherr, col="grey", main="playability")
##############
#2-6-2.R
##############
par(mfrow=c(1, 2))
hist(weatherr$playability, breaks=c(0.3, 0.4, 0.5, 0.7, 0.9), probability=FALSE,
     col="grey", main="")
hist(weatherr$playability, breaks=c(0.3, 0.4, 0.5, 0.7, 0.9), probability=TRUE,
     col="grey", main="")
##############
#2-6-3.R
##############
par(mar=c(7, 4, 4, 2))
barplot(`names<-`(ave(weatherr$playability, weatherr$outlook, weatherr$wind),
                  interaction(weatherr$outlook, weatherr$wind)),
        las=2, main="Mean playability in outlook-wind subsets")
lines(c(0, 17), rep(mean(weatherr$playability), 2), lty=2)
##############
#3-1-1.R
##############
library(dmr.claseval)
library(dmr.stats)
library(dmr.util)

library(rpart)
library(rpart.plot)
library(lattice)

data(weather, package="dmr.data")
data(weatherc, package="dmr.data")
##############
#3-2-1.R
##############
dtdat <- expand.grid(a1=seq(1, 10, 3), a2=seq(1, 10, 3))
dtdat$c <- as.factor(ifelse(dtdat$a1<=7 & dtdat$a2<=1, 1,
                            ifelse(dtdat$a1<=7 & dtdat$a2<=7, 2,
                                   ifelse(dtdat$a1<=7, 3,
                                          ifelse(dtdat$a2<=4, 4, 5)))))
  # decision tree structure
prp(rpart(c~., dtdat, minsplit=2, cp=0))
  # the corresponding domain decomposition
levelplot(c~a1*a2, dtdat, at=0.5+0:5, col.regions=gray(seq(0.1, 0.9, 0.1)),
          colorkey=list(at=0.5+0:5))
##############
#3-3-1.R
##############
data <- weather
attributes <- names(weather)[1:4]
class <- names(weather)[5]

init <- function()
{
  clabs <<- factor(levels(data[[class]]),
                   levels=levels(data[[class]]))      # class labels
  tree <<- data.frame(node=1, attribute=NA, value=NA, class=NA, count=NA,
                      `names<-`(rep(list(NA), length(clabs)),
                                paste("p", clabs, sep=".")))
  cprobs <<- (ncol(tree)-length(clabs)+1):ncol(tree)  # class probability columns
  nodemap <<- rep(1, nrow(data))
  node <<- 1
}

init()
##############
#3-3-2.R
##############
class.distribution <- function(node)
{
  tree$count[tree$node==node] <<- sum(nodemap==node)
  tree[tree$node==node,cprobs] <<- pdisc(data[nodemap==node,class])
}

class.distribution(node)
##############
#3-3-3.R
##############
class.label <- function(node)
{
  tree$class[tree$node==node] <<- which.max(tree[tree$node==node,cprobs])
}

class.label(node)
##############
#3-3-4.R
##############
maxprob <- 0.999
minsplit <- 2
maxdepth <- 8

stop.criteria <- function(node)
{
  node>=2^maxdepth || tree$count[tree$node==node]<minsplit ||
    max(tree[tree$node==node,cprobs])>maxprob
}

stop.criteria(node)
##############
#3-3-5.R
##############
weighted.impurity <- function(pd1, n1, pd0, n0, imp=entropy.p)
{
  weighted.mean(c(imp(pd1), imp(pd0)), c(n1, n0))
}

  # weighted impurity of play for outlook=overcast and outlook!=overcast
weighted.impurity(pdisc(weather$play[weather$outlook=="overcast"]),
                  sum(weather$outlook=="overcast"),
                  pdisc(weather$play[weather$outlook!="overcast"]),
                  sum(weather$outlook!="overcast"))
##############
#3-3-6.R
##############
split.eval <- function(av, sv, cl)
{
  cond <- !is.na(av) & (if (is.numeric(av)) av<=as.numeric(sv) else av==sv)

  pd1 <- pdisc(cl[cond])
  n1 <- sum(cond)
  pd0 <- pdisc(cl[!cond])
  n0 <- sum(!cond)

  if (n1>0 && n0>0)
    weighted.impurity(pd1, n1, pd0, n0, imp)
  else
    Inf
}

split.select <- function(node)
{
  splits <- data.frame()
  for (attribute in attributes)
  {
    uav <- sort(unique(data[nodemap==node,attribute]))
    if (length(uav)>1)
      splits <- rbind(splits,
                      data.frame(attribute=attribute,
                                 value=if (is.numeric(uav))
                                         midbrk(uav)
                                       else as.character(uav),
                                 stringsAsFactors=FALSE))
  }

  if (nrow(splits)>0)
    splits$eval <- sapply(1:nrow(splits),
                          function(s)
                          split.eval(data[nodemap==node,splits$attribute[s]],
                                     splits$value[s],
                                     data[nodemap==node,class]))
  if ((best.eval <- min(splits$eval))<Inf)
    tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
  best.eval
}

  # entropy-based split selection
imp <- entropy.p
split.select(node)

  # Gini index-based split selection
imp <- gini.p
split.select(node)

length(node) == 1L
##############
#3-3-7.R
##############
split.apply <- function(node)
{
  tree <<- rbind(tree,
                 data.frame(node=(2*node):(2*node+1),
                            attribute=NA, value=NA, class=NA, count=NA,
                            `names<-`(rep(list(NA), length(clabs)),
                                      paste("p", clabs, sep="."))))

  av <- data[[tree$attribute[tree$node==node]]]
  cond <- !is.na(av) & (if (is.numeric(av))
                          av<=as.numeric(tree$value[tree$node==node])
                        else av==tree$value[tree$node==node])
  nodemap[nodemap==node & cond] <<- 2*node
  nodemap[nodemap==node & !cond] <<- 2*node+1
}

split.apply(node)
##############
#3-3-8.R
##############
## a simple decision tree growing implementation
grow.dectree <- function(formula, data,
                         imp=entropy.p, maxprob=0.999, minsplit=2, maxdepth=8)
{
  init <- function()
  {
    clabs <<- factor(levels(data[[class]]),
                     levels=levels(data[[class]]))      # class labels
    tree <<- data.frame(node=1, attribute=NA, value=NA, class=NA, count=NA,
                        `names<-`(rep(list(NA), length(clabs)),
                                  paste("p", clabs, sep=".")))
    cprobs <<- (ncol(tree)-length(clabs)+1):ncol(tree)  # class probability columns
    nodemap <<- rep(1, nrow(data))
    node <<- 1
  }

  next.node <- function(node)
  {
    if (any(opn <- tree$node>node))
      min(tree$node[opn])
    else Inf
  }

  class.distribution <- function(node)
  {
    tree$count[tree$node==node] <<- sum(nodemap==node)
    tree[tree$node==node,cprobs] <<- pdisc(data[nodemap==node,class])
  }

  class.label <- function(node)
  {
    tree$class[tree$node==node] <<- which.max(tree[tree$node==node,cprobs])
  }

  stop.criteria <- function(node)
  {
    node>=2^maxdepth || tree$count[tree$node==node]<minsplit ||
      max(tree[tree$node==node,cprobs])>maxprob
  }

  split.eval <- function(av, sv, cl)
  {
    cond <- !is.na(av) & (if (is.numeric(av)) av<=as.numeric(sv) else av==sv)
    pd1 <- pdisc(cl[cond])
    n1 <- sum(cond)
    pd0 <- pdisc(cl[!cond])
    n0 <- sum(!cond)

    if (n1>0 && n0>0)
      weighted.impurity(pd1, n1, pd0, n0, imp)
    else
      Inf
  }

  split.select <- function(node)
  {
    splits <- data.frame()
    for (attribute in attributes)
    {
      uav <- sort(unique(data[nodemap==node,attribute]))
      if (length(uav)>1)
        splits <- rbind(splits,
                        data.frame(attribute=attribute,
                                   value=if (is.numeric(uav))
                                           midbrk(uav)
                                         else as.character(uav),
                                   stringsAsFactors=FALSE))
    }

    if (nrow(splits)>0)
      splits$eval <- sapply(1:nrow(splits),
                            function(s)
                            split.eval(data[nodemap==node,splits$attribute[s]],
                                       splits$value[s],
                                       data[nodemap==node,class]))
    if ((best.eval <- min(splits$eval))<Inf)
      tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
    best.eval
  }

  split.apply <- function(node)
  {
    tree <<- rbind(tree,
                   data.frame(node=(2*node):(2*node+1),
                              attribute=NA, value=NA, class=NA, count=NA,
                              `names<-`(rep(list(NA), length(clabs)),
                                        paste("p", clabs, sep="."))))

    av <- data[[tree$attribute[tree$node==node]]]
    cond <- !is.na(av) & (if (is.numeric(av))
                            av<=as.numeric(tree$value[tree$node==node])
                          else av==tree$value[tree$node==node])
    nodemap[nodemap==node & cond] <<- 2*node
    nodemap[nodemap==node & !cond] <<- 2*node+1
  }

  tree <- nodemap <- node <- NULL
  clabs <- cprobs <- NULL
  class <- y.var(formula)
  attributes <- x.vars(formula, data)

  init()
  while (is.finite(node))
  {
    class.distribution(node)
    class.label(node)
    if (!stop.criteria(node))
      if (split.select(node)<Inf)
        split.apply(node)
    node <- next.node(node)
  }
  tree$class <- clabs[tree$class]
  `class<-`(tree, "dectree")
}

## convert a dectree object to a data frame
as.data.frame.dectree <- function(x, row.names=NULL, optional=FALSE, ...)
{ as.data.frame(unclass(x), row.names=row.names, optional=optional) }

  # grow a decision tree for the weather data
tree <- grow.dectree(play~., weather)

  # grow a decision tree for the weatherc data
treec <- grow.dectree(play~., weatherc)

  # data frame conversion
as.data.frame(tree)
as.data.frame(treec)

##############
#3-4-1.R
##############
rptree <- rpart(play~., weather, minsplit=2)
##############
#3-4-2.R
##############
  # simulate cutoff of the subtree starting from the outlook=sunny split
rptree.stc <- rpart(play~., weather, subset=!(outlook %in% c("rainy", "sunny") &
                                              humidity=="high" & outlook!="sunny"),
                    minsplit=2, cp=0)

prp(rptree, varlen=0, faclen=0, main="Before subtree cutoff")
prp(rptree.stc, varlen=0, faclen=0,  main="After subtree cutoff")
##############
#3-4-3.R
##############
## transform rpart split conditions to a convenient form
rewrite.splits <- function(cond)
{
  ss <- strsplit(cond, "=")[[1]]
  attribute=ss[1]
  values <- sapply(strsplit(ss[2], ",")[[1]], deparse)
  newcond <- paste(attribute, "==", values, sep="", collapse="|")
  return(paste("(", newcond, ")", sep=""))
}

## extract a rule from an rpart tree corresponding to the path from the root
## to a given node
extract.rule <- function(rp, node)
{
  path <- path.rpart(rp, node, print.it=FALSE)[[1]][-1]
  ifelse(length(path)>0, paste(sapply(path, rewrite.splits), collapse="&"), "TRUE")
}

## calculate the error of a given node, if treated as a leaf
leaf.error <- function(rp, node, data, class)
{
  rule <- extract.rule(rp, node)
  dsub <- eval(parse(text=rule), data)
  lab <- levels(class)[rp$frame$yval[row.names(rp$frame)==node]]
  sum(lab!=class[dsub])/nrow(data[dsub,])
}

  # error of node 1, if treated as a leaf
leaf.error(rptree, 1, weather, weather$play)
  # error of node 3, which is actually a leaf
leaf.error(rptree, 3, weather, weather$play)
  # error of node 4, if treated as a leaf
leaf.error(rptree, 4, weather, weather$play)

## check whether a given node is a leaf of an rpart tree
rp.leaf <- function(rp, node)
{
  rp$frame$var[row.names(rp$frame)==node]=="<leaf>"
}

## calculate the number of instances corresponding to a node
node.card <- function(rp, node, data)
{
  rule <- extract.rule(rp, node)
  dsub <- eval(parse(text=rule), data)
  nrow(data[dsub,])
}

## calculate the error of the subtree rooted at a given node
node.error <- function(rp, node, data, class)
{
  if (rp.leaf(rp, node))
    leaf.error(rp, node, data, class)
  else
  {
    el <- node.error(rp, 2*node, data, class)
    nl <- node.card(rp, 2*node, data)
    er <- node.error(rp, 2*node+1, data, class)
    nr <- node.card(rp, 2*node+1, data)
    weighted.mean(c(el, er), c(nl, nr))
  }
}

  # error of node 1
node.error(rptree, 1, weather, weather$play)
  # error of node 3, which is actually a leaf
node.error(rptree, 3, weather, weather$play)
  # error of node 4
node.error(rptree, 1, weather, weather$play)

  # check which nodes satisfy the REP criterion
sapply(as.integer(row.names(rptree$frame)),
       function(node)
       {
         !rp.leaf(rptree, node) &&
           leaf.error(rptree, node, weather, weather$play)<=
             node.error(rptree, node, weather, weather$play)
       })
##############
#3-4-4.R
##############
## calculate the PEP error of the subtree rooted at a given node
node.pep.error <- function(rp, node, data, class)
{
  e <- node.error(rp, node, data, class)
  node <- node.card(rp, node, data)
  e1 <- (e*node+1)/(node+2)
  e + sqrt(e1*(1-e1)/node)
}

  # PEP error of node 1
node.pep.error(rptree, 1, weather, weather$play)
  # PEP error of node 4
node.pep.error(rptree, 1, weather, weather$play)

  # check which nodes would get pruned under the PEP criterion
sapply(as.integer(row.names(rptree$frame)),
       function(node)
       {
         !rp.leaf(rptree, node) &&
           leaf.error(rptree, node, weather, weather$play)<=
             node.pep.error(rptree, node, weather, weather$play)
       })
##############
#3-4-5.R
##############
## calculate the MEP error of a given node, if treated as a leaf
leaf.mep.error <- function(rp, node, data, class, m)
{
  e <- leaf.error(rp, node, data, class)
  node <- node.card(rp, node, data)
  nc <- (1-e)*node
  p <- as.double(pdisc(class)[rp$frame$yval[row.names(rp$frame)==node]])
  1-(nc+m*p)/(node+m)
}

  # MEP error of node 1, if treated as a leaf, for m=0, 2, 5
leaf.mep.error(rptree, 1, weather, weather$play, m=0)
leaf.mep.error(rptree, 1, weather, weather$play, m=2)
leaf.mep.error(rptree, 1, weather, weather$play, m=5)
  # MEP error of node 3, which is actually a leaf, for m=0, 2, 5
leaf.mep.error(rptree, 3, weather, weather$play, m=0)
leaf.mep.error(rptree, 3, weather, weather$play, m=2)
leaf.mep.error(rptree, 1, weather, weather$play, m=5)
  # MEP error of node 4, if treated as a leaf, for m=0, 2, 5
leaf.mep.error(rptree, 4, weather, weather$play, m=0)
leaf.mep.error(rptree, 4, weather, weather$play, m=2)
leaf.mep.error(rptree, 4, weather, weather$play, m=5)

## calculate the MEP error of the subtree rooted at a given node
node.mep.error <- function(rp, node, data, class, m)
{
  if (rp.leaf(rp, node))
    leaf.mep.error(rp, node, data, class, m)
  else
  {
    el <- node.mep.error(rp, 2*node, data, class, m)
    nl <- node.card(rp, 2*node, data)
    er <- node.mep.error(rp, 2*node+1, data, class, m)
    nr <- node.card(rp, 2*node+1, data)
    weighted.mean(c(el, er), c(nl, nr))
  }
}

  # MEP error of node 1 for m=0, 2, 5
node.mep.error(rptree, 1, weather, weather$play, m=0)
node.mep.error(rptree, 1, weather, weather$play, m=2)
node.mep.error(rptree, 1, weather, weather$play, m=5)
  # MEP error of node 3, which is actually a leaf, for m=0, 2, 5
node.mep.error(rptree, 3, weather, weather$play, m=0)
node.mep.error(rptree, 3, weather, weather$play, m=2)
node.mep.error(rptree, 3, weather, weather$play, m=5)
  # MEP error of node 4 for m=0, 2, 5
node.mep.error(rptree, 4, weather, weather$play, m=0)
node.mep.error(rptree, 4, weather, weather$play, m=2)
node.mep.error(rptree, 4, weather, weather$play, m=5)

  # check which nodes would get pruned under the MEP criterion
  # for m=2
sapply(as.integer(row.names(rptree$frame)),
       function(node)
       {
         !rp.leaf(rptree, node) &&
           leaf.mep.error(rptree, node, weather, weather$play, m=2)<=
             node.mep.error(rptree, node, weather, weather$play, m=2)
       })

  # for m=5
sapply(as.integer(row.names(rptree$frame)),
       function(node)
       {
         !rp.leaf(rptree, node) &&
           leaf.mep.error(rptree, node, weather, weather$play, m=5)<=
             node.mep.error(rptree, node, weather, weather$play, m=5)
       })

  # for m=10
sapply(as.integer(row.names(rptree$frame)),
       function(node)
       {
         !rp.leaf(rptree, node) &&
           leaf.mep.error(rptree, node, weather, weather$play, m=10)<=
             node.mep.error(rptree, node, weather, weather$play, m=10)
       })
##############
#3-5-1.R
##############
## decision tree prediction
predict.dectree <- function(tree, data)
{
  descend <- function(node)
  {
    if (!is.na(tree$attribute[tree$node==node]))  # unless reached a leaf
    {
      av <- data[[tree$attribute[tree$node==node]]]
      cond <- !is.na(av) & (if (is.numeric(av))
                              av<=tree$value[tree$node==node]
                            else
                              av==tree$value[tree$node==node])

      nodemap[nodemap==node & cond] <<- 2*node
      nodemap[nodemap==node & !cond] <<- 2*node+1
      descend(2*node)
      descend(2*node+1)
    }
  }

  nodemap <- rep(1, nrow(data))
  descend(1)
  tree$class[match(nodemap, tree$node)]
}

   # decision tree prediction for the weather data
predict(tree, weather)
  # decision tree prediction for the weatherc data
predict(treec, weatherc)
##############
#3-7-1.R
##############
## a simple decision tree growing implementation
## with missing value support using fractional instances
grow.dectree.frac <- function(formula, data,
                              imp=entropy.p, maxprob=0.999, minsplit=2, maxdepth=8)
{
  nmn <- function(node) { nodemap[,"node"]==node }            # nodemap entries for node node
  inn <- function(node)
  { nodemap[nodemap[,"node"]==node,"instance"] }                  # instances at node node
  wgn <- function(node) { nodemap[nodemap[,"node"]==node,"weight"] }   # weights at node node

  init <- function()
  {
    clabs <<- factor(levels(data[[class]]),
                     levels=levels(data[[class]]))      # class labels
    tree <<- data.frame(node=1, attribute=NA, value=NA, class=NA, count=NA,
                        `names<-`(rep(list(NA), length(clabs)),
                                  paste("p", clabs, sep=".")))
    cprobs <<- (ncol(tree)-length(clabs)+1):ncol(tree)  # class probability columns
    nodemap <<- cbind(instance=1:nrow(data), node=rep(1, nrow(data)),
                      weight=rep(1, nrow(data)))
    node <<- 1
  }

  next.node <- function(node)
  {
    if (any(opn <- tree$node>node))
      min(tree$node[opn])
    else Inf
  }

  class.distribution <- function(node)
  {
    tree[tree$node==node,"count"] <<- sum(wgn(node))
    tree[tree$node==node,cprobs] <<- weighted.pdisc(data[inn(node),class], w=wgn(node))
  }

  class.label <- function(node)
  {
    tree$class[tree$node==node] <<- which.max(tree[tree$node==node,cprobs])
  }

  stop.criteria <- function(node)
  {
    node>=2^maxdepth || tree[tree$node==node,"count"]<minsplit ||
      max(tree[tree$node==node,cprobs])>maxprob
  }

  split.eval <- function(av, sv, cl, w)
  {
    cond <- if (is.numeric(av)) av<=as.numeric(sv) else av==sv
    cond1 <- !is.na(av) & cond   # true split outcome
    cond0 <- !is.na(av) & !cond  # false split outcome

    pd1 <- weighted.pdisc(cl[cond1], w=w[cond1])
    n1 <- sum(w[cond1])
    pd0 <- weighted.pdisc(cl[cond0], w=w[cond0])
    n0 <- sum(w[cond0])
    pdm <- weighted.pdisc(cl[is.na(av)], w=w[is.na(av)])
    nm <- sum(w[is.na(av)])

    if (nm>0)
    {
      p1 <- if (n1+n0>0) n1/(n1+n0) else 0.5
      p0 <- 1-p1
      pd1 <- (n1*pd1 + p1*nm*pdm)/(n1+p1*nm)
      n1 <- n1 + p1*nm
      pd0 <- (n0*pd0 + p0*nm*pdm)/(n0+p0*nm)
      n0 <- n0 + nm*p0
    }

    if (n1>0 && n0>0)
      weighted.impurity(pd1, n1, pd0, n0, imp)
    else
      Inf
  }

  split.select <- function(node)
  {
    splits <- data.frame()
    for (attribute in attributes)
    {
      uav <- sort(unique(data[inn(node),attribute]))
      if (length(uav)>1)
        splits <- rbind(splits,
                        data.frame(attribute=attribute,
                                   value=if (is.numeric(uav))
                                           midbrk(uav)
                                         else as.character(uav),
                                   stringsAsFactors=FALSE))
    }

    if (nrow(splits)>0)
      splits$eval <- sapply(1:nrow(splits),
                            function(s)
                            split.eval(data[inn(node),splits$attribute[s]],
                                       splits$value[s],
                                       data[inn(node),class], wgn(node)))
    if ((best.eval <- min(splits$eval))<Inf)
      tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
    return(best.eval)
  }

  split.apply <- function(node)
  {
    tree <<- rbind(tree,
                   data.frame(node=(2*node):(2*node+1),
                              attribute=NA, value=NA, class=NA, count=NA,
                              `names<-`(rep(list(NA), length(clabs)),
                                        paste("p", clabs, sep="."))))

    av <- data[nodemap[,"instance"],tree$attribute[tree$node==node]]
    cond <- if (is.numeric(av)) av<=as.numeric(tree$value[tree$node==node])
            else av==tree$value[tree$node==node]
    cond1 <- !is.na(av) & cond   # true split outcome
    cond0 <- !is.na(av) & !cond  # false split outcome

    n1 <- sum(nodemap[nmn(node) & cond1,"weight"])
    n0 <- sum(nodemap[nmn(node) & cond0,"weight"])
    nm <- sum(nodemap[nmn(node) & is.na(av),"weight"])

    nodemap[nmn(node) & cond1,"node"] <<- 2*node
    nodemap[nmn(node) & cond0,"node"] <<- 2*node+1

    if (nm>0)
    {
      p1 <- if (n1+n0>0) n1/(n1+n0) else 0.5
      p0 <- 1-p1
      newnn <- nodemap[nmn(node) & is.na(av),,drop=FALSE]
      nodemap[nmn(node) & is.na(av),"weight"] <<-
        p1*nodemap[nmn(node) & is.na(av),"weight"]
      nodemap[nmn(node) & is.na(av),"node"] <<- 2*node
      newnn[,"weight"] <- p0*newnn[,"weight"]
      newnn[,"node"] <- 2*node+1
      nodemap <<- rbind(nodemap, newnn)
    }
  }

  tree <- cprobs <- nodemap <- node <- NULL
  clabs <- cprobs <- NULL
  class <- y.var(formula)
  attributes <- x.vars(formula, data)

  init()
  while (is.finite(node))
  {
    class.distribution(node)
    class.label(node)
    if (!stop.criteria(node))
      if (split.select(node)<Inf)
        split.apply(node)
    node <- next.node(node)
  }
  tree$class <- clabs[tree$class]
  `class<-`(tree, "dectree.frac")
}

## convert a dectree.frac object to a data frame
as.data.frame.dectree.frac <- function(x, row.names=NULL, optional=FALSE, ...)
{ as.data.frame(unclass(x), row.names=row.names, optional=optional) }

  # grow a decision tree for the weather data with missing attribute values
weatherm <- weather
weatherm$outlook[1] <- NA
weatherm$humidity[1:2] <- NA
treem <- grow.dectree.frac(play~., weatherm)

  # grow a decision tree for the weatherc data with missing attribute values
weathercm <- weather
weathercm$temperature[1:2] <- NA
weathercm$humidity[1] <- NA
treecm <- grow.dectree.frac(play~., weathercm)

  # data frame conversion
as.data.frame(treem)
as.data.frame(treecm)
##############
#3-7-2.R
##############
## decision tree prediction
## with missing value support using fractional instances
predict.dectree.frac <- function(tree, data)
{
  nmn <- function(node) { nodemap[,"node"]==node }  # nodemap entries for node node

  descend <- function(node)
  {
    if (!is.na(tree$attribute[tree$node==node]))  # unless reached a leaf
    {
      av <- data[nodemap[,"instance"],tree$attribute[tree$node==node]]
      cond <- if (is.numeric(av)) av<=as.numeric(tree$value[tree$node==node])
              else av==tree$value[tree$node==node]
      cond1 <- !is.na(av) & cond   # true split outcome
      cond0 <- !is.na(av) & !cond  # false split outcome

      nodemap[nmn(node) & cond1, "node"] <<- 2*node
      nodemap[nmn(node) & cond0, "node"] <<- 2*node+1

      if (sum(nodemap[nmn(node) & is.na(av), "weight"])>0)
      {
        n1 <- tree$count[tree$node==2*node]
        n0 <- tree$count[tree$node==2*node+1]
        p1 <- if (n1+n0>0) n1/(n1+n0) else 0.5
        p0 <- 1-p1

        newnn <- nodemap[nmn(node) & is.na(av),,drop=FALSE]
        nodemap[nmn(node) & is.na(av),"weight"] <<-
          p1*nodemap[nmn(node) & is.na(av),"weight"]
        nodemap[nmn(node) & is.na(av), "node"] <<- 2*node
        newnn[,"weight"] <- p0*newnn[,"weight"]
        newnn[,"node"] <- 2*node+1
        nodemap <<- rbind(nodemap, newnn)
      }

      descend(2*node)
      descend(2*node+1)
    }
  }

  nodemap <- cbind(instance=1:nrow(data), node=rep(1, nrow(data)),
                   weight=rep(1, nrow(data)))
  descend(1)

  clabs <- factor(levels(tree$class), levels=levels(tree$class))
  votes <- merge(nodemap, as.data.frame(tree)[,c("node", "class",
                                                 paste("p", clabs, sep="."))])
  cprobs <- (ncol(votes)-length(clabs)+1):ncol(votes)
  clabs[by(votes, votes$instance,
           function(v) which.max(colSums(v$weight*v[,cprobs])))]
}

   # decision tree prediction for the weather data with missing attribute values
predict(treem, weatherm)

  # decision tree prediction for the weatherc data with missing attribute values
predict(treecm, weathercm)
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
##############
#5-1-1.R
##############
library(dmr.claseval)
library(dmr.linreg)
library(dmr.regeval)
library(dmr.trans)
library(dmr.util)

library(lattice)

data(weatherc, package="dmr.data")
library(dmr.claseval)
library(dmr.linreg)
library(dmr.regeval)
library(dmr.trans)
library(dmr.util)

library(lattice)

data(weatherc, package="dmr.data")
data(PimaIndiansDiabetes, package="mlbench")

set.seed(12)
rpid <- runif(nrow(PimaIndiansDiabetes))
pid.train <- PimaIndiansDiabetes[rpid>=0.33,]
pid.test <- PimaIndiansDiabetes[rpid<0.33,]

set.seed(12)

  # dataset for surface plots
lcg.plot <- function(a1, a2) { 2*a1-3*a2+4 }
lcdat.plot <- `names<-`(expand.grid(seq(1, 5, 0.05), seq(1, 5, 0.05)), c("a1", "a2"))
lcdat.plot$g <- lcg.plot(lcdat.plot$a1, lcdat.plot$a2)
lcdat.plot$c <- as.factor(ustep(lcdat.plot$g))

  # datasets for parameter estimation examples
lcg <- function(a1, a2, a3, a4) { a1^2+2*a2^2-a3^2-2*a4^2+2*a1-3*a2+2*a3-3*a4+1 }
lcdat <- data.frame(a1=runif(400, min=1, max=5), a2=runif(400, min=1, max=5),
                    a3=runif(400, min=1, max=5), a4=runif(400, min=1, max=5))
lcdat$c <- as.factor(ustep(lcg(lcdat$a1, lcdat$a2, lcdat$a3, lcdat$a4)))
lcdat.train <- lcdat[1:200,]
lcdat.test <- lcdat[201:400,]

print(wf.g <- wireframe(g~a1+a2, lcdat.plot, col="grey50", zoom=0.8))
##############
#5-2-1.R
##############
  #  parameter vector for the lcg.plot function
w.plot <- c(2, -3, 4)
repf.linear(lcdat.plot[1:10,1:2], w.plot)
grad.linear(lcdat.plot[1:10,1:2], w.plot)
  # parametric model for the lcg.plot function
m.plot <- `class<-`(list(repf=repf.linear, w=w.plot), "par")
predict(m.plot, lcdat.plot[1:10,1:2])
##############
#5-2-2.R
##############
lcdat.train.lr <- lcdat.train[,1:4]
lcdat.train.lr$g <- lcg(lcdat.train$a1, lcdat.train$a2,
                        lcdat.train$a3, lcdat.train$a4)

  # "perfect" parameter vector
w.perf <- lm(g~., lcdat.train.lr)$coef[c(2:5, 1)]

  # "perfect" predictions
mse(predict.par(list(repf=repf.linear, w=w.perf), lcdat[,1:4]),
    lcg(lcdat$a1, lcdat$a2, lcdat$a3, lcdat$a4))
##############
#5-2-3.R
##############
h.t <- function(a1, a2) { ustep(lcg.plot(a1, a2)) }

lcdat.plot$h.t <- h.t(lcdat.plot$a1, lcdat.plot$a2)

wf.g.t <- wireframe(g~a1+a2, lcdat.plot, drape=TRUE, at=c(-100, 0, 100),
                    col="transparent", col.regions=c("grey30", "grey70"),
                    colorkey=FALSE, zoom=0.8)
wf.h.t <- wireframe(h.t~a1+a2, lcdat.plot, col="grey50", zoom=0.8)
l.h.t <- levelplot(h.t~a1+a2, lcdat.plot, at=c(0, 0.5, 1),
                   col.regions=c("grey30", "grey70"), colorkey=FALSE)

print(wf.g, split=c(1, 1, 2, 2), more=TRUE)
print(wf.g.t, split=c(2, 1, 2, 2), more=TRUE)
print(wf.h.t, split=c(1, 2, 2, 2), more=TRUE)
print(l.h.t, split=c(2, 2, 2, 2))
##############
#5-2-4.R
##############
## threshold representation function
repf.threshold <- function(repf) { function(data, w) ustep(repf(data, w)) }

  # "perfect" threshold model
perf.threshold <- `class<-`(list(repf=repf.threshold(repf.linear), w=w.perf), "par")
  # test set error
err(predict(perf.threshold, lcdat.test[,1:4]), lcdat.test$c)
##############
#5-2-5.R
##############
## identify a linearly separable subset of data
linsep.sub <- function(formula, data)
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  wlm <- lm(make.formula(paste("(2*as.num0(", class, ")-1)", sep=""), attributes),
            data)$coef
  wpar <- c(wlm[-1], wlm[1])  # rearrange for predict.par
  predict.par(list(repf=repf.threshold(repf.linear), w=wpar), data[,aind])==
    data[[class]]
}

  # linearly separable training and test subsets
lcdat.ls <- linsep.sub(c~., lcdat)
lcdat.train.ls <- lcdat[1:200,][lcdat.ls[1:200],]
lcdat.test.ls <- lcdat[201:400,][lcdat.ls[201:400],]
##############
#5-2-6.R
##############
logit.inv <- function(q) { (e <- exp(q))/(e+1) }

curve(logit.inv(x), from=-5, to=5, xlab="q", ylab="logit.inv(q)")

p1.lt <- function(a1, a2) { logit.inv(lcg.plot(a1, a2)) }
h.lt <- function(a1, a2) { ustep(p1.lt(a1, a2), 0.5) }

lcdat.plot$p1.lt <- p1.lt(lcdat.plot$a1, lcdat.plot$a2)
lcdat.plot$h.lt <- h.lt(lcdat.plot$a1, lcdat.plot$a2)

wf.p1.lt <- wireframe(p1.lt~a1+a2, lcdat.plot, drape=TRUE, at=c(0, 0.5, 1),
                      col="transparent", col.regions=c("grey30", "grey70"),
                      colorkey=FALSE, zoom=0.8)
wf.h.lt <- wireframe(h.lt~a1+a2, lcdat.plot, col="grey50", zoom=0.8)
l.h.lt <- levelplot(h.lt~a1+a2, lcdat.plot, at=c(-100, 0, 100),
                    col.regions=c("grey30", "grey70"), colorkey=FALSE)

print(wf.g, split=c(1, 1, 2, 2), more=TRUE)
print(wf.p1.lt, split=c(2, 1, 2, 2), more=TRUE)
print(wf.h.lt, split=c(1, 2, 2, 2), more=TRUE)
print(l.h.lt, split=c(2, 2, 2, 2))
##############
#5-2-7.R
##############
## logit representation function
repf.logit <- function(repf) { function(data, w) logit.inv(repf(data, w)) }

  # "perfect" logit model
perf.logit <- `class<-`(list(repf=repf.logit(repf.linear), w=w.perf), "par")
  # test set error
err(ustep(predict(perf.logit, lcdat.test[,1:4]), 0.5), lcdat.test$c)
  # test set loglikelihood
loglik01(predict(perf.logit, lcdat.test[,1:4]), lcdat.test$c)
##############
#5-3-1.R
##############
## threshold representation gradient
grad.threshold <- function(grad) { grad }

  # linear threshold gradient for the "perfect" parameter vector
grad.threshold(grad.linear)(lcdat.train[1:10,1:4], w.perf)
##############
#5-3-2.R
##############
## logit representation gradient
grad.logit <- function(repf, grad)
{ function(data, w) rmm(grad(data, w), (p <- repf.logit(repf)(data, w))*(1-p)) }

    # linear logit gradient for the "perfect" parameter vector
grad.logit(repf.linear, grad.linear)(lcdat.train[1:10,1:4], w.perf)
##############
#5-3-3.R
##############
## calculate parameter update based on given true and predicted values,
## gradient, and step-size using the delta rule for loglikelihood minimization
delta.loglik <- function(true.y, pred.y, gr, beta)
{
  d <- ifelse(is.finite(d <- rmm(gr, 1/(pred.y*(1-pred.y)))), d, 1)
  colSums(beta*rmm(d, true.y-pred.y))
}

## calculate parameter update based on given true and predicted values,
## gradient, and step-size using the delta rule for error minimization
delta.err <- delta.mse

  # linear threshold for the artificial data
gdl.th <- gradient.descent(c~., lcdat.train, w=rep(0, 5),
                           repf=repf.threshold(repf.linear),
                           grad=grad.threshold(grad.linear),
                           delta=delta.err, perf=err,
                           beta=1, batch=TRUE, eps=0.03)
gdl.th.ls <- gradient.descent(c~., lcdat.train.ls, w=rep(0, 5),
                              repf=repf.threshold(repf.linear),
                              grad=grad.threshold(grad.linear),
                              delta=delta.err, perf=err,
                              beta=1, batch=TRUE, eps=0.001)

  # linear logit for the artificial data
gdl.lt <- gradient.descent(c~., lcdat.train, w=rep(0, 5),
                           repf=repf.logit(repf.linear),
                           grad=grad.logit(repf.linear, grad.linear),
                           delta=delta.loglik, perf=function(p, y) -loglik01(p, y),
                           beta=0.01, batch=TRUE, eps=15.4)
gdl.lt.ls <- gradient.descent(c~., lcdat.train.ls, w=rep(0, 5),
                              repf=repf.logit(repf.linear),
                              grad=grad.logit(repf.linear, grad.linear),
                              delta=delta.loglik,
                              perf=function(p, y) -loglik01(p, y),
                              beta=0.1, batch=TRUE, eps=3)

  # linear threshold for the Pima Indians Diabetes data
pid.gdl.th <- gradient.descent(diabetes~., pid.train, w=rep(0, ncol(pid.train)),
                               repf=repf.threshold(repf.linear),
                               grad=grad.threshold(grad.linear),
                               delta=delta.err, perf=err,
                               beta=1, batch=TRUE, eps=0.28, niter=10000)

  # linear logit for the Pima Indians Diabetes data
pid.gdl.lt <- gradient.descent(diabetes~., pid.train, w=rep(0, ncol(pid.train)),
                               repf=repf.logit(repf.linear),
                               grad=grad.logit(repf.linear, grad.linear),
                               delta=delta.loglik,
                               perf=function(p, y) -loglik01(p, y),
                               beta=1e-7, batch=TRUE, eps=250, niter=1e6)

  # training set error
err(predict(gdl.th$model, lcdat.train[,1:4]), lcdat.train$c)
err(ustep(predict(gdl.lt$model, lcdat.train[,1:4]), 0.5), lcdat.train$c)

err(predict(gdl.th.ls$model, lcdat.train.ls[,1:4]), lcdat.train.ls$c)
err(ustep(predict(gdl.lt.ls$model, lcdat.train.ls[,1:4]), 0.5), lcdat.train.ls$c)

err(factor(predict(pid.gdl.th$model, pid.train[,-9]),
           levels=0:1, labels=levels(pid.train$diabetes)),
           pid.train$diabetes)
err(factor(ustep(predict(pid.gdl.lt$model, pid.train[,-9]), 0.5),
           levels=0:1, labels=levels(pid.train$diabetes)),
    pid.train$diabetes)

  # training set loglikelihood
loglik01(predict(gdl.th$model, lcdat.train[,1:4]), lcdat.train$c)
loglik01(predict(gdl.lt$model, lcdat.train[,1:4]), lcdat.train$c)

loglik01(predict(gdl.th.ls$model, lcdat.train.ls[,1:4]), lcdat.train.ls$c)
loglik01(predict(gdl.lt.ls$model, lcdat.train.ls[,1:4]), lcdat.train.ls$c)

loglik01(predict(pid.gdl.th$model, pid.train[,-9]), pid.train$diabetes)
loglik01(predict(pid.gdl.lt$model, pid.train[,-9]), pid.train$diabetes)

  # test set error
err(predict(gdl.th$model, lcdat.test[,1:4]), lcdat.test$c)
err(ustep(predict(gdl.lt$model, lcdat.test[,1:4]), 0.5), lcdat.test$c)

err(predict(gdl.th.ls$model, lcdat.test.ls[,1:4]), lcdat.test.ls$c)
err(ustep(predict(gdl.lt.ls$model, lcdat.test.ls[,1:4]), 0.5), lcdat.test.ls$c)

err(factor(predict(pid.gdl.th$model, pid.test[,-9]),
           levels=0:1, labels=levels(pid.train$diabetes)),
    pid.test$diabetes)
err(factor(ustep(predict(pid.gdl.lt$model, pid.test[,-9]), 0.5),
           levels=0:1, labels=levels(pid.train$diabetes)),
    pid.test$diabetes)

  # test set loglikelihood
loglik01(predict(gdl.th$model, lcdat.test[,1:4]), lcdat.test$c)
loglik01(predict(gdl.lt$model, lcdat.test[,1:4]), lcdat.test$c)

loglik01(predict(gdl.th.ls$model, lcdat.test.ls[,1:4]), lcdat.test.ls$c)
loglik01(predict(gdl.lt.ls$model, lcdat.test.ls[,1:4]), lcdat.test.ls$c)

loglik01(predict(pid.gdl.th$model, pid.test[,-9]), pid.test$diabetes)
loglik01(predict(pid.gdl.lt$model, pid.test[,-9]), pid.test$diabetes)
##############
#5-3-4.R
##############
## estimate linear threshold model parameters using the OLS method
ols.threshold <- function(formula, data)
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  amat <- cbind(as.matrix(data[,aind]), intercept=rep(1, nrow(data)))
  cvec <- 2*as.num0(data[[class]])-1
  `class<-`(list(repf=repf.threshold(repf.linear),
                 w=solve(t(amat)%*%amat, t(amat)%*%cvec)),
            "par")
}

  # least squares linear threshold for the artificial data
ols.th <- ols.threshold(c~., lcdat.train)
ols.th.ls <- ols.threshold(c~., lcdat.train.ls)

  # least squares linear threshold for the Pima Indians Diabetes data
pid.ols.th <- ols.threshold(diabetes~., pid.train)

  # training set error
err(predict(ols.th, lcdat.train[,1:4]), lcdat.train$c)
err(predict(ols.th.ls, lcdat.train.ls[,1:4]), lcdat.train.ls$c)
err(factor(predict(pid.ols.th, pid.train[,-9]),
           levels=0:1, labels=levels(pid.train$diabetes)),
    pid.train$diabetes)

  # test set error
err(predict(ols.th, lcdat.test[,1:4]), lcdat.test$c)
err(predict(ols.th.ls, lcdat.test.ls[,1:4]), lcdat.test.ls$c)
err(factor(predict(pid.ols.th, pid.test[,-9]),
           levels=0:1, labels=levels(pid.train$diabetes)),
    pid.test$diabetes)
##############
#5-4-1.R
##############
## estimate linear threshold model parameters using the OLS method
## for data with discrete attributes
ols.threshold.disc <- function(formula, data)
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  amat <- cbind(as.matrix(discode(~., data[,aind])), intercept=rep(1, nrow(data)))
  cvec <- 2*as.num0(data[[class]])-1
  `class<-`(list(repf=repf.disc(repf.threshold(repf.linear)),
                 w=solve(t(amat)%*%amat, t(amat)%*%cvec)),
            "par")
}

  # gradient descent for the weatherc data
w.gdl <-  gradient.descent(play~., weatherc, w=rep(0, 6),
                           repf=repf.disc(repf.threshold(repf.linear)),
                           grad=grad.disc(grad.threshold(grad.linear)),
                           delta=delta.mse, perf=err,
                           beta=1, batch=TRUE, eps=0.2)

  # OLS for the weatherc data
w.ols <- ols.threshold.disc(play~., weatherc)

  # training set error
err(factor(ustep(predict(w.gdl$model, weatherc[,1:4]), 0.5),
           levels=0:1, labels=c("no", "yes")),
    weatherc$play)
err(factor(ustep(predict(w.ols, weatherc[,1:4]), 0.5),
           levels=0:1, labels=c("no", "yes")),
    weatherc$play)
##############
#6-1-1.R
##############
library(dmr.claseval)
library(dmr.util)

library(rpart)
library(e1071)
library(ipred)

data(Vehicle, package="mlbench")

set.seed(12)
rv <- runif(nrow(Vehicle))
v.train <- Vehicle[rv>=0.33,]
v.test <- Vehicle[rv<0.33,]

  # two-class version
Vehicle01 <- Vehicle
Vehicle01$Class <- factor(ifelse(Vehicle$Class %in% c("opel", "saab"),
                                 "car", "other"))
v01.train <- Vehicle01[rv>=0.33,]
v01.test <- Vehicle01[rv<0.33,]

  # cost-insensitive decision trees
v.tree <- rpart(Class~., v.train)
v01.tree <- rpart(Class~., v01.train)

  # cost-insensitive naive Bayes classifiers
v.nb <- naiveBayes(Class~., v.train)
v01.nb <- naiveBayes(Class~., v01.train)

  # misclassification error for cost-insensitive models
v.err.b <- list(tree=err(predict(v.tree, v.test, type="c"), v.test$Class),
                nb=err(predict(v.nb, v.test), v.test$Class))

v01.err.b <- list(tree=err(predict(v01.tree, v01.test, type="c"), v01.test$Class),
                  nb=err(predict(v01.nb, v01.test), v01.test$Class))
##############
#6-2-1.R
##############
v.rm <- matrix(0, nrow=nlevels(Vehicle$Class), ncol=nlevels(Vehicle$Class),
               dimnames=list(predicted=levels(Vehicle$Class),
                             true=levels(Vehicle$Class)))

v.rm["bus","opel"] <- 7
v.rm["bus","van"] <- 0.2
v.rm["bus","saab"] <- 7
v.rm["opel","bus"] <- 1.4
v.rm["opel","saab"] <- 1
v.rm["opel","van"] <- 1.4
v.rm["saab","bus"] <- 1.4
v.rm["saab","opel"] <- 1
v.rm["saab","van"] <- 1.4
v.rm["van","bus"] <- 0.2
v.rm["van","opel"] <- 7
v.rm["van","saab"] <- 7

  # two-class version
v01.rm <- matrix(0, nrow=nlevels(Vehicle01$Class), ncol=nlevels(Vehicle01$Class),
                 dimnames=list(predicted=levels(Vehicle01$Class),
                               true=levels(Vehicle01$Class)))
v01.rm["other","car"] <- 5
v01.rm["car","other"] <- 1

  # mean  misclassification cost for cost-insensitive models
v.mc.b <- list(tree=mean.cost(predict(v.tree, v.test, type="c"), v.test$Class, v.rm),
               nb=mean.cost(predict(v.nb, v.test), v.test$Class, v.rm))

v01.mc.b <- list(tree=mean.cost(predict(v01.tree, v01.test, type="c"),
                                v01.test$Class, v01.rm),
                 nb=mean.cost(predict(v01.nb, v01.test), v01.test$Class, v01.rm))
##############
#6-2-2.R
##############
rhom2c <- function(rhom)
{
  apply(rhom, 2, sum)/(nrow(rhom)-1)
}

rhoc2m <- function(rhoc)
{
  rhom <- matrix(rhoc, nrow=length(rhoc), ncol=length(rhoc), byrow=TRUE,
                 dimnames=list(predicted=names(rhoc), true=names(rhoc)))
  `diag<-`(rhom, 0)
}

  # per-class cost vectors
v.rc <- rhom2c(v.rm)
v01.rc <- rhom2c(v01.rm)

  # per-class cost vectors in a matrix representation
v.rcm <- rhoc2m(v.rc)
v01.rcm <- rhoc2m(v01.rc)

  # misclassification cost for cost-insensitive models
  # with respect to the per-class cost vector
v.mcc.b <- list(tree=mean.cost(predict(v.tree, v.test, type="c"),
                               v.test$Class, v.rcm),
                nb=mean.cost(predict(v.nb, v.test), v.test$Class, v.rcm))
##############
#6-3-1.R
##############
## generate an instance-weighting cost-sensitive wrapper
mc.weight <- function(alg, predf=predict)
{
  wrapped.alg <- function(formula, data, rho, ...)
  {
    class <- y.var(formula)
    w <- rho[data[[class]]]
    do.call(alg, list(formula, data, weights=w, ...))
  }

  list(alg=wrapped.alg, predict=predf)
}

  # weighting wrapper around rpart
rpart.w <- mc.weight(rpart, predf=function(...) predict(..., type="c"))

  # decision trees with instance weighting
v.tree.w <- rpart.w$alg(Class~., v.train, v.rc)
v01.tree.w <- rpart.w$alg(Class~., v01.train, v01.rc)

  # mean misclassification cost with respect to the cost matrix
v.mc.w <- list(tree=mean.cost(rpart.w$predict(v.tree.w, v.test), v.test$Class, v.rm))
v01.mc.w <- list(tree=mean.cost(rpart.w$predict(v01.tree.w, v01.test),
                                v01.test$Class, v01.rm))

# mean misclassification cost with respect to the per-class cost vector
v.mcc.w <- list(tree=mean.cost(rpart.w$predict(v.tree.w, v.test),
                               v.test$Class, v.rcm))

  # misclassification error
v.err.w <- list(tree=err(rpart.w$predict(v.tree.w, v.test), v.test$Class))
v01.err.w <- list(tree=err(rpart.w$predict(v01.tree.w, v01.test), v01.test$Class))

  # confusion matrix
confmat(rpart.w$predict(v.tree.w, v.test), v.test$Class)
confmat(rpart.w$predict(v01.tree.w, v01.test), v01.test$Class)
##############
#6-3-2.R
##############
## generate an instance-resampling cost-sensitive wrapper
mc.resample <- function(alg, predf=predict)
{
  wrapped.alg <- function(formula, data, rho, ...)
  {
    class <- y.var(formula)
    w <- rho[data[[class]]]
    rs <- na.omit(c(rep(1:nrow(data), floor(w)),
                    ifelse(runif(nrow(data))<=w-floor(w), 1:nrow(data), NA)))
    do.call(alg, list(formula, data[rs,], ...))
  }

  list(alg=wrapped.alg, predict=predf)
}

  # resampling wrapper around rpart
rpart.s <- mc.resample(rpart, predf=function(...) predict(..., type="c"))

  # resampling wrapper around naiveBayes
naiveBayes.s <- mc.resample(naiveBayes)

  # decision trees with instance resampling
v.tree.s <- rpart.s$alg(Class~., v.train, v.rc)
v01.tree.s <- rpart.s$alg(Class~., v01.train, v01.rc)

  # naive Bayes with instance resampling
v.nb.s <- naiveBayes.s$alg(Class~., v.train, v.rc)
v01.nb.s <- naiveBayes.s$alg(Class~., v01.train, v01.rc)

  # mean misclassification cost with respect to the cost matrix
v.mc.s <- list(tree=mean.cost(rpart.s$predict(v.tree.s, v.test), v.test$Class, v.rm),
               nb=mean.cost(naiveBayes.s$predict(v.nb.s, v.test),
                            v.test$Class, v.rm))
v01.mc.s <- list(tree=mean.cost(rpart.s$predict(v01.tree.s, v01.test),
                                v01.test$Class, v01.rm),
                 nb=mean.cost(naiveBayes.s$predict(v01.nb.s, v01.test),
                              v01.test$Class, v01.rm))

# mean misclassification cost with respect to the per-class cost vector
v.mcc.s <- list(tree=mean.cost(rpart.s$predict(v.tree.s, v.test),
                               v.test$Class, v.rcm),
                nb=mean.cost(naiveBayes.s$predict(v.nb.s, v.test),
                             v.test$Class, v.rcm))

  # misclassification error
v.err.s <- list(tree=err(rpart.s$predict(v.tree.s, v.test), v.test$Class),
                nb=err(naiveBayes.s$predict(v.nb.s, v.test), v.test$Class))
v01.err.s <- list(tree=err(rpart.s$predict(v01.tree.s, v01.test), v01.test$Class),
                  nb=err(naiveBayes.s$predict(v01.nb.s, v01.test), v01.test$Class))

  # confusion matrix
confmat(rpart.s$predict(v.tree.s, v.test), v.test$Class)
confmat(naiveBayes.s$predict(v.nb.s, v.test), v.test$Class)

confmat(rpart.s$predict(v01.tree.s, v01.test), v01.test$Class)
confmat(naiveBayes.s$predict(v01.nb.s, v01.test), v01.test$Class)
##############
#6-3-3.R
##############
## minimum-cost rule for a single instance
mincostclas1 <- function(p, rho)
{ factor(which.min(rho%*%p), levels=1:length(p), labels=names(p)) }

## minimum-cost rule for multiple instances
mincostclas <- function(p, rho) { apply(p, 1, mincostclas1, rho) }

## generate a minimum-cost wrapper
mc.mincost <- function(alg, ppredf=predict)
{
  wrapped.alg <- function(formula, data, rho=NULL, ...)
  {
    list(model=alg(formula, data, ...), rho=rho)
  }

  wrapped.predict <- function(model, data, rho=NULL, ...)
  {
    mincostclas(ppredf(model$model, data, ...),
                if (is.null(rho)) model$rho else rho)
  }

  list(alg=wrapped.alg, predict=wrapped.predict)
}

  # minimum-cost wrapper around rpart
rpart.m <- mc.mincost(rpart)

  # minimum-cost wrapper around naiveBayes
naiveBayes.m <- mc.mincost(naiveBayes, ppredf=function(...) predict(..., type="r"))

  # decision trees with minimum-cost prediction
v.tree.m <- rpart.m$alg(Class~., v.train, v.rm, cp=0.025)
v01.tree.m <- rpart.m$alg(Class~., v01.train, v01.rm, cp=0.025)

  # naive Bayes with minimum-cost prediction
v.nb.m <- naiveBayes.m$alg(Class~., v.train, v.rm)
v01.nb.m <- naiveBayes.m$alg(Class~., v01.train, v01.rm)

  # mean misclassification cost with respect to the cost matrix
v.mc.m <- list(tree=mean.cost(rpart.m$predict(v.tree.m, v.test), v.test$Class, v.rm),
               nb=mean.cost(naiveBayes.m$predict(v.nb.m, v.test),
                            v.test$Class, v.rm))
v01.mc.m <- list(tree=mean.cost(rpart.m$predict(v01.tree.m, v01.test),
                                v01.test$Class, v01.rm),
                 nb=mean.cost(naiveBayes.m$predict(v01.nb.m, v01.test),
                              v01.test$Class, v01.rm))

# mean misclassification cost with respect to the per-class cost vector
v.mcc.m <- list(tree=mean.cost(rpart.m$predict(v.tree.m, v.test),
                               v.test$Class, v.rcm),
                nb=mean.cost(naiveBayes.m$predict(v.nb.m, v.test),
                             v.test$Class, v.rcm))

  # misclassification error
v.err.m <- list(tree=err(rpart.m$predict(v.tree.m, v.test), v.test$Class),
                nb=err(naiveBayes.m$predict(v.nb.m, v.test), v.test$Class))

v01.err.m <- list(tree=err(rpart.m$predict(v01.tree.m, v01.test), v01.test$Class),
                  nb=err(naiveBayes.m$predict(v01.nb.m, v01.test), v01.test$Class))

  # confusion matrix
confmat(rpart.m$predict(v.tree.m, v.test), v.test$Class)
confmat(naiveBayes.m$predict(v.nb.m, v.test), v.test$Class)

confmat(rpart.m$predict(v01.tree.m, v01.test), v01.test$Class)
confmat(naiveBayes.m$predict(v01.nb.m, v01.test), v01.test$Class)
##############
#6-3-4.R
##############
## generate an instance-relabeling cost-sensitive wrapper
mc.relabel <- function(alg, palg=alg, pargs=NULL, predf=predict, ppredf=predict)
{
  wrapped.alg <- function(formula, data, rho, ...)
  {
    class <- y.var(formula)
    model <- do.call(palg, c(list(formula, data), pargs))
    prob <- ppredf(model, data)
    data[[class]] <- mincostclas(prob, rho)
    alg(formula, data, ...)
  }

  list(alg=wrapped.alg, predict=predf)
}

  # relabeling wrapper around rpart
rpart.l <- mc.relabel(rpart, pargs=list(cp=0.025),
                      predf=function(...) predict(..., type="c"))

  # relabeling wrapper around rpart using bagging for probability estimation
rpart.bagg.l <- mc.relabel(rpart, bagging,
                           pargs=list(control=rpart.control(cp=0.025)),
                           predf=function(...) predict(..., type="c"),
                           ppredf=function(...) predict(..., type="p",
                                                             aggregation="a"))

  # relabeling wrapper around naiveBayes
naiveBayes.l <- mc.relabel(naiveBayes, ppredf=function(...) predict(..., type="r"))

  # decision trees with instance relabeling
v.tree.l <- rpart.l$alg(Class~., v.train, v.rm)
v.tree.bagg.l <- rpart.bagg.l$alg(Class~., v.train, v.rm)
v01.tree.l <- rpart.l$alg(Class~., v01.train, v01.rm)
v01.tree.bagg.l <- rpart.bagg.l$alg(Class~., v01.train, v01.rm)

  # naive Bayes with instance relabeling
v.nb.l <- naiveBayes.l$alg(Class~., v.train, v.rm)
v01.nb.l <- naiveBayes.l$alg(Class~., v01.train, v01.rm)

  # mean misclassification cost with respect to the cost matrix
v.mc.l <- list(tree=mean.cost(rpart.l$predict(v.tree.l, v.test), v.test$Class, v.rm),
               tree.bagg=mean.cost(rpart.bagg.l$predict(v.tree.bagg.l, v.test),
                                   v.test$Class, v.rm),
               nb=mean.cost(naiveBayes.l$predict(v.nb.l, v.test),
                            v.test$Class, v.rm))
v01.mc.l <- list(tree=mean.cost(rpart.l$predict(v01.tree.l, v01.test),
                                v01.test$Class, v01.rm),
                 tree.bagg=mean.cost(rpart.bagg.l$predict(v01.tree.bagg.l, v01.test),
                                     v01.test$Class, v01.rm),
                 nb=mean.cost(naiveBayes.l$predict(v01.nb.l, v01.test),
                              v01.test$Class, v01.rm))

# mean misclassification cost with respect to the per-class cost vector
v.mcc.l <- list(tree=mean.cost(rpart.l$predict(v.tree.l, v.test),
                               v.test$Class, v.rcm),
                tree.bagg=mean.cost(rpart.bagg.l$predict(v.tree.bagg.l, v.test),
                                    v.test$Class, v.rcm),
                nb=mean.cost(naiveBayes.l$predict(v.nb.l, v.test),
                             v.test$Class, v.rcm))

  # misclassification error
v.err.l <- list(tree=err(rpart.l$predict(v.tree.l, v.test), v.test$Class),
                tree.bagg=err(rpart.bagg.l$predict(v.tree.bagg.l, v.test),
                              v.test$Class),
                nb=err(naiveBayes.l$predict(v.nb.l, v.test), v.test$Class))
v01.err.l <- list(tree=err(rpart.l$predict(v01.tree.l, v01.test), v01.test$Class),
                  tree.bagg=err(rpart.bagg.l$predict(v01.tree.bagg.l, v01.test),
                                v01.test$Class),
                  nb=err(naiveBayes.l$predict(v01.nb.l, v01.test), v01.test$Class))

  # confusion matrix
confmat(rpart.l$predict(v.tree.l, v.test), v.test$Class)
confmat(rpart.bagg.l$predict(v.tree.bagg.l, v.test), v.test$Class)
confmat(naiveBayes.l$predict(v.nb.l, v.test), v.test$Class)

confmat(rpart.l$predict(v01.tree.l, v01.test), v01.test$Class)
confmat(naiveBayes.l$predict(v01.nb.l, v01.test), v01.test$Class)
##############
#6-4-1.R
##############
  # mean misclassification cost with respect to the cost matrix
v.mc <- c(tree=v.mc.b$tree,
          tree.w=v.mc.w$tree,
          tree.s=v.mc.s$tree,
          tree.m=v.mc.m$tree,
          tree.l=v.mc.l$tree,
          tree.bagg.l=v.mc.l$tree.bagg,
          nb=v.mc.b$nb,
          nb.s=v.mc.s$nb,
          nb.m=v.mc.m$nb,
          nb.l=v.mc.l$nb)

v01.mc <- c(tree=v01.mc.b$tree,
            tree.w=v01.mc.w$tree,
            tree.s=v01.mc.s$tree,
            tree.m=v01.mc.m$tree,
            tree.l=v01.mc.l$tree,
            tree.bagg.l=v01.mc.l$tree.bagg,
            nb=v01.mc.b$nb,
            nb.s=v01.mc.s$nb,
            nb.m=v01.mc.m$nb,
            nb.l=v01.mc.l$nb)

  # mean misclassification cost with respect to the per-class cost vector
v.mcc <- c(tree=v.mcc.b$tree,
           tree.w=v.mcc.w$tree,
           tree.s=v.mcc.s$tree,
           tree.m=v.mcc.m$tree,
           tree.l=v.mcc.l$tree,
           tree.bagg.l=v.mcc.l$tree.bagg,
           nb=v.mcc.b$nb,
           nb.s=v.mcc.s$nb,
           nb.m=v.mcc.m$nb,
           nb.l=v.mcc.l$nb)

  # misclassification error
v.err <- c(tree=v.err.b$tree,
           tree.w=v.err.w$tree,
           tree.s=v.err.s$tree,
           tree.m=v.err.m$tree,
           tree.l=v.err.l$tree,
           tree.bagg.l=v.err.l$tree.bagg,
           nb=v.err.b$nb,
           nb.s=v.err.s$nb,
           nb.m=v.err.m$nb,
           nb.l=v.err.l$nb)

v01.err <- c(tree=v01.err.b$tree,
             tree.w=v01.err.w$tree,
             tree.s=v01.err.s$tree,
             tree.m=v01.err.m$tree,
             tree.l=v01.err.l$tree,
             tree.bagg.l=v01.err.l$tree.bagg,
             nb=v01.err.b$nb,
             nb.s=v01.err.s$nb,
             nb.m=v01.err.m$nb,
             nb.l=v01.err.l$nb)


barplot(v.mc, ylab="Mean cost (matrix)", las=2)
lines(c(0, 12), rep(v.mc[1], 2), lty=2)
lines(c(0, 12), rep(v.mc[7], 2), lty=3)

barplot(v.mcc, ylab="Mean cost (per-class)", las=2)
lines(c(0, 12), rep(v.mcc[1], 2), lty=2)
lines(c(0, 12), rep(v.mcc[7], 2), lty=3)

barplot(v.err, ylab="Error", las=2)
lines(c(0, 12), rep(v.err[1], 2), lty=2)
lines(c(0, 12), rep(v.err[7], 2), lty=3)

barplot(v01.mc, ylab="Mean cost (two-class)", las=2)
lines(c(0, 12), rep(v01.mc[1], 2), lty=2)
lines(c(0, 12), rep(v01.mc[7], 2), lty=3)

barplot(v01.err, ylab="Error (two-class)", las=2)
lines(c(0, 12), rep(v01.err[1], 2), lty=2)
lines(c(0, 12), rep(v01.err[7], 2), lty=3)
##############
#6-5-1.R
##############
## run misclassification costs experiments
mc.experiment <- function(algs, datasets, classes, rmax=10, m=25, k=10, node=1,
                          palgs=NULL, args.c=NULL, args.p=NULL, pargs=NULL,
                          predfs=predict, ppredfs.algs=predict,
                          ppredfs.palgs=predict)
{
  crossval.rs <- function(rs, ...) { .Random.seed <<- rs; crossval(...) }

  results <- NULL
  for (dn in 1:length(datasets))
  {
    res <- NULL

    data <- get(datasets[dn])
    class <- classes[dn]
    formula <- make.formula(class, ".")

    for (i in 1:m)
    {
      rho <- matrix(round(runif(nlevels(data[[class]])^2, min=1, max=rmax)-0.5),
                    nrow=nlevels(data[[class]]),
                    ncol=nlevels(data[[class]]),
                    dimnames=list(predicted=levels(data[[class]]),
                                  true=levels(data[[class]])))
      diag(rho) <- 0
      rhoc <- rhom2c(rho)

      for (an in 1:length(algs))
      {
        alg <- get(algs[[an]])
        palg <- if (!is.null(palgs[[an]])) get(palgs[[an]])
        arg.c <- args.c[[an]]
        arg.p <- args.p[[an]]
        parg <- pargs[[an]]
        predf <- if (is.vector(predfs)) predfs[[an]] else predfs
        ppredf.alg <- if (is.vector(ppredfs.algs)) ppredfs.algs[[an]]
                      else ppredfs.algs
        ppredf.palg <- if (is.vector(ppredfs.palgs)) ppredfs.palgs[[an]]
                       else ppredfs.palgs
        aname <- paste(algs[an], palgs[an], sep=".")

        if (is.null(palg))
        {
          alg.w <- mc.weight(alg, predf)
          alg.s <- mc.resample(alg, predf)
          alg.m <- mc.mincost(alg, ppredf.alg)
          alg.l <- mc.relabel(alg, pargs=arg.p, predf=predf, ppredf=ppredf.alg)
        }
        else
          alg.lp <- mc.relabel(alg, palg, pargs=parg,
                               predf=predf, ppredf=ppredf.palg)

        rs <- .Random.seed
        cv.b <- crossval(alg, formula, data, args=arg.c, predf=predf, k=k, n=n)
        if (is.null(palg))
        {
          cv.w <- crossval.rs(rs, alg.w$alg, formula, data,
                              args=c(list(rhoc), arg.c),
                              predf=alg.w$predict, k=k, n=n)
          cv.s <- crossval.rs(rs, alg.s$alg, formula, data,
                              args=c(list(rhoc), arg.c),
                              predf=alg.s$predict, k=k, n=n)
          cv.m <- crossval.rs(rs, alg.m$alg, formula, data,
                              args=c(list(rho), arg.p),
                              predf=alg.m$predict, k=k, n=n)
          cv.l <- crossval.rs(rs, alg.l$alg, formula, data,
                              args=c(list(rho), arg.c),
                              predf=alg.l$predict, k=k, n=n)
          mc <- data.frame(b=mean.cost(cv.b$pred, cv.b$true, rho),
                           w=mean.cost(cv.w$pred, cv.w$true, rho),
                           s=mean.cost(cv.s$pred, cv.s$true, rho),
                           m=mean.cost(cv.m$pred, cv.m$true, rho),
                           l=mean.cost(cv.l$pred, cv.l$true, rho))
          mc$d.w <- (mc$b-mc$w)/mc$b
          mc$d.s <- (mc$b-mc$s)/mc$b
          mc$d.m <- (mc$b-mc$m)/mc$b
          mc$d.l <- (mc$b-mc$l)/mc$b
          e <- data.frame(b=err(cv.b$pred, cv.b$true),
                          w=err(cv.w$pred, cv.w$true),
                          s=err(cv.s$pred, cv.s$true),
                          m=err(cv.m$pred, cv.m$true),
                          l=err(cv.l$pred, cv.l$true))
        }
        else
        {
          cv.lp <- crossval.rs(rs, alg.lp$alg, formula, data,
                               args=c(list(rho), arg.c),
                               predf=alg.lp$predict, k=k, n=n)
          mc <- data.frame(b=mean.cost(cv.b$pred, cv.b$true, rho),
                           lp=mean.cost(cv.lp$pred, cv.lp$true, rho))
          e <- data.frame(b=err(cv.b$pred, cv.b$true),
                          lp=err(cv.lp$pred, cv.lp$true))
          mc$d.lp <- (mc$b-mc$lp)/mc$b
        }

        res[[aname]]$mc <- rbind(res[[aname]]$mc, mc)
        res[[aname]]$e <- rbind(res[[aname]]$e, e)
      }
    }
    results <- c(results, list(res))
  }
  `names<-`(results, datasets)
}

  # experiments with decision trees and naive Bayes
  # for the Vehicle and Vehicle01 datasets
mc.res <- mc.experiment(c("rpart", "rpart", "naiveBayes"),
                        c("Vehicle", "Vehicle01"), c("Class", "Class"),
                        palgs=list(NULL, "bagging", NULL),
                        args.p=list(list(cp=0.025), list(cp=0.025), NULL),
                        pargs=list(NULL, list(control=rpart.control(cp=0.025)),
                                   NULL),
                        predfs=c(function(...) predict(..., type="c"),
                                 function(...) predict(..., type="c"), predict),
                        ppredfs.algs=c(predict, predict,
                                       function(...) predict(..., type="r")),
                        ppredfs.palgs=list(NULL,
                                           function(...) predict(..., type="p",
                                                                 aggregation="a"),
                                           NULL))

barplot(colMeans(cbind(mc.res$Vehicle$rpart.NULL$mc[,6:9],
                       mc.res$Vehicle$rpart.bagging$mc[,3])),
        main="Four-class, rpart", ylab="Cost reduction",
        las=2, ylim=c(-0.01, 0.11),
        names.arg=c("weight", "resample", "mincost", "relabel", "relabel.b"))
barplot(colMeans(mc.res$Vehicle$naiveBayes.NULL$mc[,7:9]),
        main="Four-class, naiveBayes", ylab="Cost reduction",
        las=2, ylim=c(-0.01, 0.11),
        names.arg=c("resample", "mincost", "relabel"))

barplot(colMeans(cbind(mc.res$Vehicle01$rpart.NULL$mc[,6:9],
                       mc.res$Vehicle01$rpart.bagging$mc[,3])),
        main="Two-class, rpart", ylab="Cost reduction",
        las=2, ylim=c(-0.26, 0.15),
        names.arg=c("weight", "resample", "mincost", "relabel", "relabel.b"))
barplot(colMeans(mc.res$Vehicle01$naiveBayes.NULL$mc[,7:9]),
        main="Two-class, naiveBayes", ylab="Cost reduction",
        las=2, ylim=c(-0.26, 0.15),
        names.arg=c("resample", "mincost", "relabel"))
##############
#7-1-1.R
##############
library(dmr.util)
library(rpart)

#data(Soybean, package="mlbench")
pairs(iris[1:4], main = "Edgar Anderson's Iris Data", pch = 21, bg = c("red", "green3", "blue")[unclass(iris$Species)])
mdata<-iris
set.seed(12)
rs <- runif(nrow(mdata))
s.train <- mdata[rs>=0.33,]
s.test <- mdata[rs<0.33,]

s.tree <- rpart(Species~., s.train)
##############
#7-2-10.R
##############
  # predicted scores and true class labels
sctab <- data.frame(score=c(1, 1, 3, 4, 5, 5, 6, 7, 9, 9),
                    class=factor(c(0, 0, 1, 0, 0, 1, 1 ,0, 1, 1)))

  # operating point identification
scroc <- t(sapply(c(sort(unique(sctab$score)), Inf),
                  function(sc)
                  {
                    pred <- factor(as.numeric(sctab$score<sc),
                                   levels=levels(sctab$class))
                    list(tpr=tpr(cm <- confmat(pred, sctab$class)),
                         fpr=fpr(cm))
                  }))

  # ROC curve
plot(scroc, type="l", xlab="FP rate", ylab="TP rate")
##############
#7-2-11.R
##############
roc <- function(pred.s, true.y)
{
  cutoff <- Inf  # start with all instances classified as negative
  tp <- fp <- 0
  tn <- sum(2-as.integer(true.y))  # all negative instances
  fn <- sum(as.integer(true.y)-1)  # all positive instances
  rt <- data.frame()

  sord <- order(pred.s, decreasing=TRUE)  # score ordering
  for (i in 1:length(sord))
  {
    if (pred.s[sord[i]] < cutoff)
    {
      rt <- rbind(rt, data.frame(tpr=tp/(tp+fn), fpr=fp/(fp+tn), cutoff=cutoff))
      cutoff <- pred.s[sord[i]]
    }

    p <- as.integer(true.y[sord[i]])-1  # next positive classified as positive
    n <- 2-as.integer(true.y[sord[i]])  # next negative classified as positive
    tp <- tp+p
    fp <- fp+n
    tn <- tn-n
    fn <- fn-p
  }
  rt <- rbind(rt, data.frame(tpr=tp/(tp+fn), fpr=fp/(fp+tn), cutoff=cutoff))
}
s01.tree<-s.tree
s01.test<-s.test
prp(s01.tree, varlen=0, faclen=0)
plotcp(s01.tree)
plot(s01.tree, uniform=TRUE,main="Decision tree")
pred_labels1<-predict(s01.tree, s01.test)
#finding a name of factor level
levels(s01.test$Species)[which.is.max(pred_labels1[1,])]
#converting factor to the string vector
sapply(s01.test$Species, as.character)
#converting factor to the factor index vector
fit1=c();for(i in 1:nrow(pred_labels1))fit1=c(fit1,which.is.max(pred_labels1[i,])); fit1
#converting factor to index 
#levels(s01.test$Species)<-seq_len(length(levels(s01.test$Species)))
as.integer(s01.test$Species)
  # ROC curve for the decision tree model
s01.roc <- roc(fit1, s01.test$Species)
plot(s01.roc$fpr, s01.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")

  # ROC curve for a random model
s01rand <- runif(nrow(s01.test))
s01rand.roc <- roc(s01rand, s01.test$Species)
lines(s01rand.roc$fpr, s01rand.roc$tpr, lty=2)
##############
#7-2-12.R
##############
## assign class labels according to the given cutoff value
cutclass <- function(s, cutoff, labels) { factor(ustep(s, cutoff), labels=labels) }

## identify the best cutoff value
## satisfying the minimum tpr or maximum fpr constraint
roc.shift <- function(r, min.tpr=NULL, max.fpr=NULL)
{
  if (!is.null(min.tpr))
    max(r$cutoff[r$tpr>=min.tpr])
  else if (!is.null(max.fpr))
    min(r$cutoff[r$fpr<=max.fpr])
  else
    0.5
}

s01.tree<-s.tree
s01.test<-s.test
s01.labels<-'a'
  # shift to achieve tpr>0.85 at minimum fpr
s01.t085 <- roc.shift(s01.roc, min.tpr=0.85)
s01.cm.t085 <- confmat(cutclass(predict(s01.tree, s01.test)[,2],
                                s01.t085, s01.labels),
                       s01.test$Species)
  # shift to achieve maximum tpr at fpr<0.5
s01.f05 <- roc.shift(s01.roc, max.fpr=0.5)
s01.cm.f05 <- confmat(cutclass(predict(s01.tree, s01.test)[,2], s01.f05, s01.labels),
                      s01.test$Species)
  # the ROC curve
plot(s01.roc$fpr, s01.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
  # the default operating point
points(fpr(s01.cm), tpr(s01.cm), pch=8)
  # the shifted operating points
points(fpr(s01.cm.t085), tpr(s01.cm.t085), pch=1)
points(fpr(s01.cm.f05), tpr(s01.cm.f05), pch=2)
##############
#7-2-13.R
##############
mixclass <- function(c1, c2, p)
{ factor(ifelse(p<runif(length(c1)), c1, c2), labels=levels(c1)) }

  # interpolate between the two shifted operating points
s01.mix <- mixclass(cutclass(predict(s01.tree, s01.test)[,2], s01.t085, s01.labels),
                    cutclass(predict(s01.tree, s01.test)[,2], s01.f05, s01.labels),
                    0.75)
s01.cmi <- confmat(s01.mix, s01.test$Class)

  # the ROC curve
plot(s01.roc$fpr, s01.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
  # the default operating point
points(fpr(s01.cm), tpr(s01.cm), pch=8)
  # the 1st shifted operating point
points(fpr(s01.cm.t085), tpr(s01.cm.t085), pch=1)
  # the 2nd shifted operating point
points(fpr(s01.cm.f05), tpr(s01.cm.f05), pch=2)
  # the interpolated operating point
points(fpr(s01.cmi), tpr(s01.cmi), pch=5)
##############
#7-2-14.R
##############
auc <- function(roc)
{ n <- nrow(roc); sum((roc$tpr[1:n-1]+roc$tpr[2:n])*diff(roc$fpr)/2) }

  # area under the ROC curve for the decision tree model
auc(s01.roc)
  # area under the ROC curve for a random model
auc(s01rand.roc)
##############
#7-2-15.R
##############
wroc <- function(pred.s, true.y, w=rep(1, length(true.y)))
{
  cutoff <- Inf  # start with all instances classified as negative
  tp <- fp <- 0
  tn <- sum((2-as.integer(true.y))*w)  # all negative instances
  fn <- sum((as.integer(true.y)-1)*w)  # all positive instances
  rt <- data.frame()

  sord <- order(pred.s, decreasing=TRUE)  # score ordering
  for (i in 1:length(sord))
  {
    if (pred.s[sord[i]] < cutoff)
    {
      rt <- rbind(rt, data.frame(tpr=tp/(tp+fn), fpr=fp/(fp+tn), cutoff=cutoff))
      cutoff <- pred.s[sord[i]]
    }

    p <- (as.integer(true.y[sord[i]])-1)*w[sord[i]]  # next positive
    n <- (2-as.integer(true.y[sord[i]]))*w[sord[i]]  # next negative
    tp <- tp+p
    fp <- fp+n
    tn <- tn-n
    fn <- fn-p
  }
  rt <- rbind(rt, data.frame(tpr=tp/(tp+fn), fpr=fp/(fp+tn), cutoff=cutoff))
}

  # ROC curve with double weight for the brown-spot class
s01.w1roc <- wroc(predict(s01.tree, s01.test)[,2], s01.test$Species, s01.w1test)
plot(s01.w1roc$fpr, s01.w1roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
auc(s01.w1roc)

  # ROC curve with 10 times less weight for instances with plant.stand=1
s01.w2roc <- wroc(predict(s01.tree, s01.test)[,2], s01.test$Species, s01.w2test)
lines(s01.w2roc$fpr, s01.w2roc$tpr, lty=2)
legend("bottomright", c("brown-spot x2", "plant.stand=1 x10"), lty=1:2)
auc(s01.w2roc)
##############
#7-2-16.R
##############
## likelihood for probabilistic classifier evaluation
## assuming eps for 0 probabilities
lik <- function(prob.y, true.y, eps=.Machine$double.eps)
{
  prod(pmax(sapply(1:length(tn <- as.numeric(true.y)),
                   function(i) prob.y[i,tn[i]]), eps))
}

## likelihood for probabilistic binary classifier evaluation
## assuming eps for 0 probabilities
lik01 <- function(prob.y, true.y, eps=.Machine$double.eps)
{
  prod((py <- pmin(pmax(prob.y, eps), 1-eps))^(t01 <- as.num0(true.y))*(1-py)^(1-t01))
}

  # likelihood for the Soybean data
lik(predict(s.tree, s.test), s.test$Species)
lik(predict(s01.tree, s01.test), s01.test$Species)
lik01(predict(s01.tree, s01.test)[,2], s01.test$Species)
##############
#7-2-17.R
##############
## loglikelihood for probabilistic classifier evaluation
## assuming eps for 0 probabilities
loglik <- function(prob.y, true.y, eps=.Machine$double.eps)
{
  sum(log(pmax(sapply(1:length(tn <- as.numeric(true.y)),
                      function(i) prob.y[i,tn[i]]), eps)))
}

## loglikelihood for probabilistic binary classifier evaluation
## assuming eps for 0 probabilities
loglik01 <- function(prob.y, true.y, eps=.Machine$double.eps)
{
  sum((t01 <- as.num0(true.y))*log(py <- pmin(pmax(prob.y, eps), 1-eps))+
      (1-t01)*log(1-py))
}

  # loglikelihood for the Soybean data
loglik(predict(s.tree, s.test), s.test$Class)
loglik(predict(s01.tree, s01.test), s01.test$Class)
loglik01(predict(s01.tree, s01.test)[,2], s01.test$Class)
##############
#7-2-1.R
##############
err <- function(pred.y, true.y) { mean(pred.y!=true.y) }

err(predict(s.tree, s.test, type="c"), s.test$Class)
##############
#7-2-2.R
##############
werr <- function(pred.y, true.y, w=rep(1, length(true.y)))
{ weighted.mean(pred.y!=true.y, w) }

  # double weight for the least frequent class
s.w2test <- ifelse(s.test$Class=="herbicide-injury", 2, 1)
werr(predict(s.tree, s.test, type="c"), s.test$Class, s.w2test)

  # random per-class weights 1..5
s.wctest <- round(runif(nlevels(s.test$Class), min=1, max=5))
s.w3test <- s.wctest[s.test$Class]
werr(predict(s.tree, s.test, type="c"), s.test$Class, s.w3test)
##############
#7-2-3.R
##############
mean.cost <- function(pred.y, true.y, rho) { mean(diag(rho[pred.y,true.y])) }

  # uniform cost matrix
s.r1test <- matrix(1, nrow=nlevels(s.test$Class), ncol=nlevels(s.test$Class))
diag(s.r1test) <- 0
mean.cost(predict(s.tree, s.test, type="c"), s.test$Class, s.r1test)

  # double cost for misclassifying the least frequent class
s.r2test <- matrix(1, nrow=nlevels(s.test$Class), ncol=nlevels(s.test$Class))
s.r2test[,levels(s.test$Class)=="herbicide-injury"] <- 2
diag(s.r2test) <- 0
mean.cost(predict(s.tree, s.test, type="c"), s.test$Class, s.r2test)
  # this should give the same result
sum(s.w2test)/nrow(s.test)*werr(predict(s.tree, s.test, type="c"),
                                s.test$Class, s.w2test)

  # random per-class costs 1..5
s.r3test <- matrix(s.wctest, nrow=nlevels(s.test$Class), ncol=nlevels(s.test$Class),
                   byrow=TRUE)
diag(s.r3test) <- 0
mean.cost(predict(s.tree, s.test, type="c"), s.test$Class, s.r3test)
  # this should give the same result
sum(s.w3test)/nrow(s.test)*werr(predict(s.tree, s.test, type="c"),
                                s.test$Class, s.w3test)

  # random costs 1..5
s.r4test <- matrix(round(runif(nlevels(s.test$Class)*nlevels(s.test$Class),
                               min=1, max=5)),
                   nrow=nlevels(s.test$Class), ncol=nlevels(s.test$Class))
diag(s.r4test) <- 0
mean.cost(predict(s.tree, s.test, type="c"), s.test$Class, s.r4test)
##############
#7-2-4.R
##############
confmat <- function(pred.y, true.y)
{ table(pred.y, true.y, dnn=c("predicted", "true")) }

s.cm <- confmat(predict(s.tree, s.test, type="c"), s.test$Class)
  # error
(sum(s.cm)-sum(diag(s.cm)))/(sum(s.cm))
  # mean misclassification cost
sum(s.cm*s.r4test)/sum(s.cm)
##############
#7-2-5.R
##############
s01.labels <- c("other", "brown-spot")
Soybean01 <- Soybean
Soybean01$Class <- factor(ifelse(Soybean$Class=="brown-spot", "brown-spot", "other"),
                          levels=s01.labels)

s01.train <- Soybean01[rs>=0.33,]
s01.test <- Soybean01[rs<0.33,]

s01.tree <- rpart(Class~., s01.train)
##############
#7-2-6.R
##############
tpr <- function(cm) { if (is.nan(p <- cm[2,2]/(cm[2,2]+cm[1,2]))) 1 else p }
fpr <- function(cm) { if (is.nan(p <- cm[2,1]/(cm[2,1]+cm[1,1]))) 1 else p }
precision <- function(cm) { if (is.nan(p <- cm[2,2]/(cm[2,2]+cm[2,1]))) 1 else p }
recall <- tpr
sensitivity <- tpr
specificity <- function(cm) { if (is.nan(p <- cm[1,1]/(cm[2,1]+cm[1,1]))) 1 else p }

s01.cm <- confmat(predict(s01.tree, s01.test, type="c"), s01.test$Class)

list(tpr=tpr(s01.cm),
     fpr=fpr(s01.cm),
     precision=precision(s01.cm),
     recall=recall(s01.cm),
     sensitivity=sensitivity(s01.cm),
     specificity=specificity(s01.cm))
##############
#7-2-7.R
##############
f.measure <- function(cm) { 1/mean(c(1/precision(cm), 1/recall(cm))) }

f.measure(s01.cm)
##############
#7-2-8.R
##############
## per-class 1 vs. rest confusion matrices
confmat01 <- function(pred.y, true.y)
{
  `names<-`(lapply(levels(true.y),
                   function(d)
                   {
                     cm <- confmat(factor(as.integer(pred.y==d), levels=0:1),
                                   factor(as.integer(true.y==d), levels=0:1))
                   }), levels(true.y))
}

s.cm01 <- confmat01(predict(s.tree, s.test, type="c"), s.test$Species)
  # average TP rate, FP rate, and f-measure
rowMeans(sapply(s.cm01, function(cm) c(tpr=tpr(cm), fpr=fpr(cm), fm=f.measure(cm))))
##############
#7-2-9.R
##############
wconfmat <- function(pred.y, true.y, w=rep(1, length(true.y)))
{ weighted.table(pred.y, true.y, w=w, dnn=c("predicted", "true")) }

  # double weight for the brown-spot class
s01.w1test <- ifelse(s01.test$Class=="brown-spot", 2, 1)
s01.w1cm <- wconfmat(predict(s01.tree, s01.test, type="c"),
                     s01.test$Class, s01.w1test)
tpr(s01.w1cm)
fpr(s01.w1cm)

  # 10 times less weight for instances with plant.stand=1
s01.w2test <- ifelse(!is.na(s01.test$plant.stand) & s01.test$plant.stand=="1",
                     0.1, 1)
s01.w2cm <- wconfmat(predict(s01.tree, s01.test, type="c"),
                     s01.test$Class, s01.w2test)
tpr(s01.w2cm)
fpr(s01.w2cm)
##############
#7-3-1.R
##############
holdout <- function(alg, formula, data, args=NULL, predf=predict, prob=FALSE,
                    p=0.33, n=1)
{
  yn <- as.character(formula)[2]  # class column name
  ylabs <- levels(data[[yn]])     # class labels
  pred.y <- NULL  # predictions
  true.y <- NULL  # true class labels

  for (t in 1:n)
  {
    r <- runif(nrow(data))
    train <- data[r>=p,]
    test <- data[r<p,]
    model <- do.call(alg, c(list(formula, train), args))
    pred.y <- c(pred.y, predf(model, test))
    true.y <- c(true.y, test[[yn]])
  }

  if (!is.null(ylabs))
  {
    if (!prob)
      pred.y <- factor(pred.y, levels=1:length(ylabs), labels=ylabs)
    true.y <- factor(true.y, levels=1:length(ylabs), labels=ylabs)
  }

  return(data.frame(pred=pred.y, true=true.y))
}

  # hold-out evaluation of discrete predictions
s01ho <- holdout(rpart, Class~., Soybean01,
                 predf=function(...) predict(..., type="c"), n=10)
err(s01ho$pred, s01ho$true)
confmat(s01ho$pred, s01ho$true)

  # hold-out evaluation of probabilistic predictions
s01hop <- holdout(rpart, Class~., Soybean01,
                  predf=function(...) predict(..., type="p")[,2], prob=TRUE, n=10)
s01hop.roc <- roc(s01hop$pred, s01hop$true)
plot(s01hop.roc$fpr, s01hop.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
auc(s01hop.roc)
##############
#7-3-2.R
##############
crossval <- function(alg, formula, data, args=NULL, predf=predict, prob=FALSE,
                     k=10, n=1)
{
  yn <- as.character(formula)[2]  # class column name
  ylabs <- levels(data[[yn]])     # class labels
  pred.y <- NULL  # predictions
  true.y <- NULL  # true class labels

  for (t in 1:n)
  {
    ind <- sample(k, size=nrow(data), replace=TRUE)  # index of k random subsets
    for (i in 1:k)
    {
      train <- data[ind!=i,]
      test <- data[ind==i,]
      model <- do.call(alg, c(list(formula, train), args))
      pred.y <- c(pred.y, predf(model, test))
      true.y <- c(true.y, test[[yn]])
    }
  }

  if (!is.null(ylabs))
  {
    if (!prob)
      pred.y <- factor(pred.y, levels=1:length(ylabs), labels=ylabs)
    true.y <- factor(true.y, levels=1:length(ylabs), labels=ylabs)
  }

  return(data.frame(pred=pred.y, true=true.y))
}

  # 3-fold cross-validation for discrete predictions
s01cv3 <- crossval(rpart, Class~., Soybean01,
                   predf=function(...) predict(..., type="c"), k=3)
err(s01cv3$pred, s01cv3$true)
confmat(s01cv3$pred, s01cv3$true)

  # 10-fold cross-validation for discrete predictions
s01cv10 <- crossval(rpart, Class~., Soybean01,
                    predf=function(...) predict(..., type="c"), k=10)
err(s01cv10$pred, s01cv10$true)
confmat(s01cv10$pred, s01cv10$true)

  # 20-fold cross-validation for discrete predictions
s01cv20 <- crossval(rpart, Class~., Soybean01,
                    predf=function(...) predict(..., type="c"), k=20)
err(s01cv20$pred, s01cv20$true)
confmat(s01cv20$pred, s01cv20$true)

  # 4x5-fold cross-validation for discrete predictions
s01cv4x5 <- crossval(rpart, Class~., Soybean01,
                     predf=function(...) predict(..., type="c"), k=5, n=4)
err(s01cv20$pred, s01cv4x5$true)
confmat(s01cv4x5$pred, s01cv4x5$true)

  # 10-fold cross-validation for probabilistic predictions
s01cv10p <- crossval(rpart, Class~., Soybean01,
                     predf=function(...) predict(..., type="p")[,2], prob=TRUE, k=10)
s01cv10p.roc <- roc(s01cv10p$pred, s01cv10p$true)
plot(s01cv10p.roc$fpr, s01cv10p.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
auc(s01cv10p.roc)
##############
#7-3-3.R
##############
leave1out <- function(alg, formula, data, args=NULL, predf=predict, prob=FALSE)
{
  yn <- as.character(formula)[2]  # class column name
  ylabs <- levels(data[[yn]])     # class labels
  pred.y <- NULL  # predictions
  true.y <- NULL  # true class labels

  for (i in 1:nrow(data))
  {
    train <- data[-i,]
    test <- data[i,]
    model <- do.call(alg, c(list(formula, train), args))
    pred.y <- c(pred.y, predf(model, test))
    true.y <- c(true.y, test[[yn]])
  }

  if (!is.null(ylabs))
  {
    if (!prob)
      pred.y <- factor(pred.y, levels=1:length(ylabs), labels=ylabs)
    true.y <- factor(true.y, levels=1:length(ylabs), labels=ylabs)
  }

  return(data.frame(pred=pred.y, true=true.y))
}

  # leave-one-out for discrete predictions
s01l1o <- leave1out(rpart, Class~., Soybean01,
                    predf=function(...) predict(..., type="c"))
err(s01l1o$pred, s01l1o$true)
confmat(s01l1o$pred, s01l1o$true)

  # leave-one-out for probabilistic predictions
s01l1op <- leave1out(rpart, Class~., Soybean01,
                     predf=function(...) predict(..., type="p")[,2], prob=TRUE)
s01l1op.roc <- roc(s01l1op$pred, s01l1op$true)
plot(s01l1op.roc$fpr, s01l1op.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
auc(s01l1op.roc)
##############
#7-3-4.R
##############
bootstrap <- function(alg, formula, data, args=NULL, predf=predict, prob=FALSE,
                      w=0.632, m=100)
{
  yn <- as.character(formula)[2]  # class column name
  ylabs <- levels(data[[yn]])     # class labels
  pred.y.w <- NULL  # predictions
  true.y.w <- NULL  # true class labels

  for (t in 1:m)
  {
    bag <- sample(nrow(data), size=nrow(data), replace=TRUE)
    train <- data[bag,]
    test <- data[-bag,]
    model <- do.call(alg, c(list(formula, train), args))
    pred.y.w <- c(pred.y.w, predf(model, test))
    true.y.w <- c(true.y.w, test[[yn]])
  }

  if (w<1)
  {
    model <- do.call(alg, c(list(formula, data), args))
    pred.y.1w <- predf(model, data)
    true.y.1w <- data[[yn]]
    w <- c(rep(w/m, length(pred.y.w)), rep(1-w, nrow(data)))
  }
  else
  {
    pred.y.1w <- true.y.1w <- NULL
    w <- rep(w/m, length(pred.y.w))
  }

  pred.y <- c(pred.y.w, pred.y.1w)
  true.y <- c(true.y.w, true.y.1w)

  if (!is.null(ylabs))
  {
    if (!prob)
      pred.y <- factor(pred.y, levels=1:length(ylabs), labels=ylabs)
    true.y <- factor(true.y, levels=1:length(ylabs), labels=ylabs)
  }

  return(data.frame(pred=pred.y, true=true.y, w=w))
}

  # 20x bootstrap for discrete predictions
s01bs20 <- bootstrap(rpart, Class~., Soybean01,
                     predf=function(...) predict(..., type="c"), w=1, m=20)
err(s01bs20$pred, s01bs20$true)
confmat(s01bs20$pred, s01bs20$true)

  # 20x .632 bootstrap for discrete predictions
s01.632bs20 <- bootstrap(rpart, Class~., Soybean01,
                         predf=function(...) predict(..., type="c"), m=20)
  # weighted error
werr(s01.632bs20$pred, s01.632bs20$true, s01.632bs20$w)
  # weighted confusion matrix
wconfmat(s01.632bs20$pred, s01.632bs20$true, s01.632bs20$w)

  # 20x bootstrap for probabilistic predictions
s01bs20p <- bootstrap(rpart, Class~., Soybean01,
                      predf=function(...) predict(..., type="p")[,2], prob=TRUE,
                      w=1, m=20)
s01bs20p.roc <- roc(s01bs20p$pred, s01bs20p$true)
plot(s01bs20p.roc$fpr, s01bs20p.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
auc(s01bs20p.roc)

  # 20x .632 bootstrap for probabilistic predictions
s01.632bs20p <- bootstrap(rpart, Class~., Soybean01,
                          predf=function(...) predict(..., type="p")[,2], prob=TRUE,
                          m=20)
  # weighted ROC
s01.632bs20p.roc <- wroc(s01.632bs20p$pred, s01.632bs20p$true, s01.632bs20p$w)
lines(s01.632bs20p.roc$fpr, s01.632bs20p.roc$tpr, lty=2)
legend("bottomright", c("plain", ".632"), lty=1:2)
auc(s01.632bs20p.roc)
##############
#7-3-5.R
##############
eval.bias.var <- function(alg, formula, data, args=NULL, predf=predict,
                          perf=err, wperf=werr, p=0.66, n=100)
{
  yn <- as.character(formula)[2]  # class column name
  performance <- data.frame()
  for (i in 1:n)
  {
    r <- runif(nrow(data))
    data.avail <- data[r<p,]   # pretend this is the available dataset
    data.new <- data[r>=0.7,]  # pretend this a new dataset
    model <- do.call(alg, c(list(formula, data.avail), args))

    cv3 <- crossval(alg, formula, data.avail, args=args, predf=predf, k=3)
    cv5 <- crossval(alg, formula, data.avail, args=args, predf=predf, k=5)
    cv10 <- crossval(alg, formula, data.avail, args=args, predf=predf, k=10)
    cv20 <- crossval(alg, formula, data.avail, args=args, predf=predf, k=20)
    cv5x4 <- crossval(alg, formula, data.avail, args=args, predf=predf, k=5, n=4)
    ho <- holdout(alg, formula, data.avail, args=args, predf=predf)
    hox10 <- holdout(alg, formula, data.avail, args=args, predf=predf, n=10)
    l1o <- leave1out(alg, formula, data.avail, args=args, predf=predf)
    bs10 <- bootstrap(alg, formula, data.avail, args=args, predf=predf, w=1, m=10)
    bs50 <- bootstrap(alg, formula, data.avail, args=args, predf=predf, w=1, m=50)
    bs10.632 <- bootstrap(alg, formula, data.avail, args=args, predf=predf, m=10)
    bs50.632 <- bootstrap(alg, formula, data.avail, args=args, predf=predf, m=50)

    performance <- rbind(performance,
                         data.frame(perf(predf(model, data.new), data.new[[yn]]),
                                    perf(cv3$pred, cv3$true),
                                    perf(cv5$pred, cv5$true),
                                    perf(cv10$pred, cv10$true),
                                    perf(cv20$pred, cv20$true),
                                    perf(cv5x4$pred, cv5x4$true),
                                    perf(ho$pred, ho$true),
                                    perf(hox10$pred, hox10$true),
                                    perf(l1o$pred, l1o$true),
                                    perf(bs10$pred, bs10$true),
                                    perf(bs50$pred, bs50$true),
                                    wperf(bs10.632$pred, bs10.632$true, bs10.632$w),
                                    wperf(bs50.632$pred, bs50.632$true, bs50.632$w)))
  }

  names(performance) <- c("true", "3-CV", "5-CV", "10-CV", "20-CV", "4x5-CV",
                          "HO", "10xHO", "L1O", "10-BS", "50-BS",
                          "10-BS.632", "50-BS.632")
  bias <- apply(performance[,-1]-performance[,1], 2, mean)
  variance <- apply(performance[,-1], 2, var)

  list(performance=performance, bias=bias, variance=variance)
}

  # the commented lines run a 200-repetition experiment, which takes a long time
#s01.ebv <- eval.bias.var(rpart, Class~., Soybean01,
#                         predf=function(...) predict(..., type="c"), n=200)
  # this can be used for a quick illustration
s01.ebv <- eval.bias.var(rpart, Class~., Soybean01,
                         predf=function(...) predict(..., type="c"), n=10)

boxplot(s01.ebv$performance[,-1], main="Error", las=2)
lines(c(0, 13), rep(mean(s01.ebv$performance[,1]), 2))
barplot(s01.ebv$bias, main="Bias", las=2)
barplot(s01.ebv$variance, main="Variance", las=2)
##############
#8-1-1.R
##############
library(dmr.regeval)
library(dmr.trans)
library(dmr.util)

data(weatherr, package="dmr.data")
data(BostonHousing, package="mlbench")

set.seed(12)
rbh <- runif(nrow(BostonHousing))
bh.train <- BostonHousing[rbh>=0.33,-4]
bh.test <- BostonHousing[rbh<0.33,-4]

set.seed(12)

  # generate artificial data
lrdat <- data.frame(a1=floor(runif(400, min=1, max=5)),
                    a2=floor(runif(400, min=1, max=5)),
                    a3=floor(runif(400, min=1, max=5)),
                    a4=floor(runif(400, min=1, max=5)))
lrdat$f1 <- 3*lrdat$a1+4*lrdat$a2-2*lrdat$a3+2*lrdat$a4-3
lrdat$f2 <- tanh(lrdat$f1/10)
lrdat$f3 <- lrdat$a1^2+2*lrdat$a2^2-lrdat$a3^2-2*lrdat$a4^2+
            2*lrdat$a1-3*lrdat$a2+2*lrdat$a3-3*lrdat$a4+1
lrdat$f4 <- 2*tanh(lrdat$a1-2*lrdat$a2+3*lrdat$a3-lrdat$a4+1)-
            3*tanh(-2*lrdat$a1+3*lrdat$a2-2*lrdat$a3+lrdat$a4-1)+2

  # training and test subsets
lrdat.train <- lrdat[1:200,]
lrdat.test <- lrdat[201:400,]
##############
#8-2-1.R
##############
## parametric regression prediction for a given model and dataset
predict.par <-  function(model, data) { model$repf(data, model$w) }

  # perfect representation function for f4
repf.perf4 <- function(data, w)
{
  w[2*(n <- ncol(data))+3]*tanh(rowSums(cmm(data, w[1:n]))+w[n+1]) +
    w[2*n+4]*tanh(rowSums(cmm(data, w[(n+2):(2*n+1)]))+w[2*n+2]) + w[2*n+5]
}

  # perfect parameters for f4
w.perf4 <- c(1, -2, 3, -1, 1, -2, 3, -2, 1, -1, 2, -3, 2)
  # perfect model for f4
mod.perf4 <- `class<-`(list(w=w.perf4, repf=repf.perf4), "par")
  # test set error
mse(predict(mod.perf4, lrdat.test[,1:4]), lrdat.test$f4)
##############
#8-2-2.R
##############
## linear representation function
repf.linear <- function(data, w)
{ rowSums(cmm(data, w[1:(n <- ncol(data))])) + w[n+1] }

  # perfect parameter vector for f1
w.perf1 <- c(3, 4, -2, 2, -3)
  # perfect model for f1
mod.perf1 <- `class<-`(list(w=w.perf1, repf=repf.linear), "par")
  # test set error
mse(predict(mod.perf1, lrdat.test[,1:4]), lrdat.test$f1)
##############
#8-3-1.R
##############
## gradient of the linear representation function
grad.linear <- function(data, w) { cbind(data, 1) }

  # gradient for the first 10 instances
grad.linear(lrdat.train[1:10,1:4], rep(0, 5))
##############
#8-3-2.R
##############
## calculate parameter update based on given true and predicted values,
## gradient, and step-size using the delta rule for MSE minimization
delta.mse <- function(true.y, pred.y, gr, beta)
{ colSums(beta*rmm(gr, (true.y-pred.y))) }

  # parameter updates for the perfect model for f1
delta.mse(lrdat.train$f1, predict(mod.perf1, lrdat.train[,1:4]),
          grad.linear(lrdat.train[,1:4], w.perf1), 0.1)
  # parameter updates for the perfect model for f1
  # with modified target function values
delta.mse(lrdat.train$f1+0.1, predict(mod.perf1, lrdat.train[,1:4]),
          grad.linear(lrdat.train[,1:4], w.perf1), 0.1)
##############
#8-3-3.R
##############
## perform gradient descent iterative parameter estimation
## for parametric regression models
gradient.descent <- function(formula, data, w, repf, grad, delta=delta.mse, perf=mse,
                             beta=0.001, batch=FALSE, randomize=!batch,
                             eps=0.001, niter=1000)
{
  f <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes
  true.y <- as.num0(data[[f]])
  model <- `class<-`(list(repf=repf, w=w), "par")
  iter <- 0

  repeat
  {
    if (batch)
    {
      pred.y <- predict.par(model, data[,aind])
      model$w <- model$w + delta(true.y, pred.y, grad(data[,aind], model$w), beta)
    }
    else
    {
      pred.y <- numeric(nrow(data))
      xind <- if (randomize) sample.int(nrow(data)) else 1:nrow(data)
      for (i in 1:length(xind))
      {
        av <- data[xind[i], aind]
        pred.y[xind[i]] <- predict.par(model, av)
        model$w <- model$w +
                   delta(true.y[xind[i]], pred.y[xind[i]], grad(av, model$w), beta)
      }
    }
    iter <- iter+1

    cat("iteration ", iter, ":\t", p <- perf(pred.y, true.y), "\n")
    if (p < eps || iter >= niter)
      return(list(model=model, perf=p))
  }
}

  # linear model for f1
gd1 <- gradient.descent(f1~a1+a2+a3+a4, lrdat.train, w=rep(0, 5),
                        repf=repf.linear, grad=grad.linear, beta=0.01, eps=0.0001)

  # linear model for the Boston Housing data
bh.gd <- gradient.descent(medv~., bh.train, w=rep(0, ncol(bh.train)),
                          repf=repf.linear, grad=grad.linear, beta=1e-6, eps=25,
                          niter=5000)

  # test set error
mse(predict(gd1$model, lrdat.test[,1:4]), lrdat.test$f1)
mse(predict(bh.gd$model, bh.test[,-13]), bh.test$medv)
##############
#8-3-4.R
##############
## estimate linear model parameters using the OLS method
ols <- function(formula, data)
{
  f <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  amat <- cbind(as.matrix(data[,aind]), intercept=rep(1, nrow(data)))
  fvec <- data[[f]]
  `class<-`(list(repf=repf.linear, w=solve(t(amat)%*%amat, t(amat)%*%fvec)), "par")
}

  # linear model for f1
ols1 <- ols(f1~a1+a2+a3+a4, lrdat.train)
  # linear model for the Boston Housing data
bh.ols <- ols(medv~., bh.train)

  # test set error
mse(predict(ols1, lrdat.test[,1:4]), lrdat.test$f1)
mse(predict(bh.ols, bh.test[,-13]), bh.test$medv)
##############
#8-4-1.R
##############
## representation function wrapper to handle discrete attributes
repf.disc <- function(repf)
{ function(data, w) { repf(discode(~., data, b=c(-1,1)), w) } }

## representation function gradient wrapper to handle discrete attributes
grad.disc <- function(grad)
{ function(data, w) { grad(discode(~., data, b=c(-1,1)), w) } }

## estimate linear model parameters using the OLS method
## with discrete attributes
ols.disc <- function(formula, data)
{
  f <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  amat <- cbind(as.matrix(discode(~., data[,aind], b=c(-1,1))),
                intercept=rep(1, nrow(data)))
  fvec <- data[[f]]
  `class<-`(list(repf=repf.disc(repf.linear),
                 w=solve(t(amat)%*%amat, t(amat)%*%fvec)), "par")
}

  # gradient descent for the weatherr data
w.gdl <- gradient.descent(playability~., weatherr, w=c(rep(0, 5), 1),
                          repf=repf.disc(repf.linear), grad=grad.disc(grad.linear),
                          beta=0.0001, eps=0.005)
mse(weatherr$playability, predict(w.gdl$model, weatherr[,1:4]))

  # OLS for the weatherr data
w.ols <- ols.disc(playability~., weatherr)
mse(predict(w.ols, weatherr[,1:4]), weatherr$playability)
##############
#8-6-1.R
##############
## generalized representation function
repf.gen <- function(link.inv, repf=repf.linear)
{ function(data, w) { link.inv(repf(data, w)) } }

## generalized representation function gradient
grad.gen <- function(link.inv.deriv, repf=repf.linear, grad=grad.linear)
{ function(data, w) { rmm(grad(data, w), link.inv.deriv(repf(data, w))) } }

  # perfect inverse link function for f2
link2.inv <- function(v) { tanh(v/10) }
  # and its derivative
link2.inv.deriv <- function(v) { (1-tanh(v/10)^2)/10 }

  # perfect generalized linear representation function for f2
repf.gen2 <- repf.gen(link2.inv)
  # and its gradient
grad.gen2 <- grad.gen(link2.inv.deriv)

  # gradient descent estimation of generalized linear model parameters for f2
gd2g <- gradient.descent(f2~a1+a2+a3+a4, lrdat.train, w=rep(0, 5),
                         repf=repf.gen2, grad=grad.gen2,
                         beta=0.5, eps=0.0001)
  # test set error
mse(predict(gd2g$model, lrdat.test[,1:4]), lrdat.test$f2)
##############
#8-6-2.R
##############
## a naive application of OLS to a generalized linear representation
ols.gen <- function(formula, data, link=function(v) v, link.inv=function(v) v)
{
  f <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  amat <- cbind(as.matrix(data[,aind]), intercept=rep(1, nrow(data)))
  fvec <- link(data[[f]])
  `class<-`(list(repf=repf.gen(link.inv), w=solve(t(amat)%*%amat, t(amat)%*%fvec)),
            "par")
}

  # perfect link function for f2
link2 <- function(v) { 10*atanh(v) }

  # estimate of generalized linear model parameters for f2
ols2g <- ols.gen(f2~a1+a2+a3+a4, lrdat.train, link=link2, link.inv=link2.inv)
  # test set error
mse(predict(ols2g, lrdat.test[,1:4]), lrdat.test$f2)
##############
#8-6-3.R
##############
## enhanced representation function
repf.enh <- function(enhance, repf=repf.linear)
{ function(data, w) { repf(enhance(data), w) } }

## enhanced representation function gradient
grad.enh <- function(enhance, grad=grad.linear)
{ function(data, w) { grad(enhance(data), w) } }

## estimate linear model parameters using the OLS method
## with enhanced representation
ols.enh <- function(formula, data, enhance=function(data) data)
{
  f <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  amat <- cbind(as.matrix(enhance(data[,aind])), intercept=rep(1, nrow(data)))
  fvec <- data[[f]]
  `class<-`(list(repf=repf.enh(enhance), w=solve(t(amat)%*%amat, t(amat)%*%fvec)),
            "par")
}

  # perfect representation enhancement for f3
enhance3 <- function(data) { cbind(data, sq=data^2) }

  # gradient descent estimation for f3
gd3e <- gradient.descent(f3~a1+a2+a3+a4, lrdat.train, w=rep(0, 9),
                         repf=repf.enh(enhance3), grad=grad.enh(enhance3),
                         beta=0.001, eps=0.05)
  # test set error
mse(predict(gd3e$model, lrdat.test[,1:4]), lrdat.test$f3)

  # ols estimation for f3
ols3e <- ols.enh(f3~a1+a2+a3+a4, lrdat.train, enhance3)
  # test set error
mse(predict(ols3e, lrdat.test[,1:4]), lrdat.test$f3)
##############
#8-6-4.R
##############
## polynomial representation enhancement
enhance.poly <- function(data, p=2)
{ do.call(cbind, lapply(1:p, function(j) data^j)) }

## polynomial regression representation function
repf.poly <- function(p=2)
{ repf.enh(function(data) enhance.poly(data, p), repf.linear) }

## polynomial regression representation function gradient
grad.poly <- function(p=2)
{ grad.enh(function(data) enhance.poly(data, p), grad.linear) }

  # gradient descent polynomial regression estimation for f3
gd3p <- gradient.descent(f3~a1+a2+a3+a4, lrdat.train, w=rep(0, 9),
                         repf=repf.poly(p=2), grad=grad.poly(p=2),
                         beta=0.001, eps=0.005)
  # test set error
mse(predict(gd3p$model, lrdat.test[,1:4]), lrdat.test$f3)

  # OLS polynomial regression estimation for f3
ols3p <- ols.enh(f3~a1+a2+a3+a4, lrdat.train, enhance.poly)
  # test set error
mse(predict(ols3p, lrdat.test[,1:4]), lrdat.test$f3)
##############
#9-1-1.R
##############
library(dmr.regeval)
library(dmr.stats)
library(dmr.util)

library(rpart)

data(weatherr, package="dmr.data")
##############
#9-2-1.R
##############
  # example target function
rtf <- function(a1, a2) { sin(a1+a2)/(a1+a2) }

  # artificial dataset
rtdat <- data.frame(a1=floor(runif(300, min=1, max=6)),
                    a2=floor(runif(300, min=1, max=6)))
rtdat$f <- rtf(rtdat$a1, rtdat$a2)

  # regression tree
rtf.rp <- rpart(f~., rtdat)
  # target function predictions
rtf.p <- function(a1, a2) { predict(rtf.rp, data.frame(a1, a2)) }

  # 3D plots
par(mfrow=1:2, mar=rep(0.1, 4))
a1 <- a2 <- seq(1, 5, 0.1)
  # true f
persp(a1, a2, outer(a1, a2, rtf), zlab="true f", theta=30, phi=30, col="grey")
  # predicted f
persp(a1, a2, outer(a1, a2, rtf.p), zlab="predicted f", theta=30, phi=30, col="grey")
##############
#9-3-1.R
##############
data <- weatherr
attributes <- names(weatherr)[1:4]
target <- names(weatherr)[5]

init <- function()
{
  tree <<- data.frame(node=1, attribute=NA, value=NA, target=NA,
                      count=NA, mean=NA, variance=NA)
  nodemap <<- rep(1, nrow(data))
  node <<- 1
}

init()
##############
#9-3-2.R
##############
target.summary <- function(node)
{
  tree$count[tree$node==node] <<- sum(nodemap==node)
  tree$mean[tree$node==node] <<- mean(data[nodemap==node,target])
  tree$variance[tree$node==node] <<- var1(data[nodemap==node,target])
}

target.summary(node)
##############
#9-3-3.R
##############
target.value <- function(node)
{
  tree$target[tree$node==node] <<- tree$mean[tree$node==node]
}

target.value(node)
##############
#9-3-4.R
##############
minvar <- 0.005
minsplit <- 2
maxdepth <- 8

stop.criteria <- function(node)
{
  node>=2^maxdepth || tree$count[tree$node==node]<minsplit ||
    tree$variance[tree$node==node]<minvar
}

stop.criteria(node)
##############
#9-3-5.R
##############
weighted.dispersion <- function(v1, v0, w1, w0, disp=var1)
{
  if (missing(w1) || missing(w0))
    weighted.mean(c(disp(v1), disp(v0)), c(length(v1), length(v0)))
  else
    weighted.mean(c(disp(v1, w1), disp(v0, w0)), c(length(v1), length(v0)))
}

  # weighted dispersion of playability for outlook=overcast and outlook!=overcast
weighted.dispersion(weatherr$playability[weatherr$outlook=="overcast"],
                    weatherr$playability[weatherr$outlook!="overcast"])
##############
#9-3-6.R
##############
split.eval <- function(av, sv, tv)
{
  cond <- !is.na(av) & (if (is.numeric(av)) av<=as.numeric(sv) else av==sv)
  v1 <- tv[cond]
  n1 <- sum(cond)
  v0 <- tv[!cond]
  n0 <- sum(!cond)
  if (n1>0 && n0>0)
    weighted.dispersion(v1, v0)
  else
    Inf
}

split.select <- function(node)
{
  splits <- data.frame()
  for (attribute in attributes)
  {
    uav <- sort(unique(data[nodemap==node,attribute]))
    if (length(uav)>1)
      splits <- rbind(splits,
                      data.frame(attribute=attribute,
                                 value=if (is.numeric(uav))
                                         midbrk(uav)
                                       else as.character(uav),
                                 stringsAsFactors=FALSE))
  }

  if (nrow(splits)>0)
    splits$eval <- sapply(1:nrow(splits),
                          function(s)
                          split.eval(data[nodemap==node,splits$attribute[s]],
                                     splits$value[s],
                                     data[nodemap==node,target]))
  if ((best.eval <- min(splits$eval))<Inf)
    tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
  best.eval
}

  # variance-based split selection
split.select(node)
##############
#9-3-7.R
##############
split.apply <- function(node)
{
  tree <<- rbind(tree,
                 data.frame(node=(2*node):(2*node+1), attribute=NA, value=NA, target=NA,
                            count=NA, mean=NA, variance=NA))

  av <- data[[tree$attribute[tree$node==node]]]
  cond <- !is.na(av) & (if (is.numeric(av))
                          av<=as.numeric(tree$value[tree$node==node])
                        else av==tree$value[tree$node==node])
  nodemap[nodemap==node & cond] <<- 2*node
  nodemap[nodemap==node & !cond] <<- 2*node+1
}

split.apply(node)
##############
#9-3-8.R
##############
## a simple regression tree growing implementation
grow.regtree <- function(formula, data, minvar=0.005, minsplit=2, maxdepth=8)
{
  init <- function()
  {
    tree <<- data.frame(node=1, attribute=NA, value=NA, target=NA,
                        count=NA, mean=NA, variance=NA)
    nodemap <<- rep(1, nrow(data))
    node <<- 1
  }

  next.node <- function(node)
  {
    if (any(opn <- tree$node>node))
      min(tree$node[opn])
    else Inf
  }

  target.summary <- function(node)
  {
    tree$count[tree$node==node] <<- sum(nodemap==node)
    tree$mean[tree$node==node] <<- mean(data[nodemap==node,target])
    tree$variance[tree$node==node] <<- var1(data[nodemap==node,target])
  }

  target.value <- function(node)
  {
    tree$target[tree$node==node] <<- tree$mean[tree$node==node]
  }

  stop.criteria <- function(node)
  {
    node>=2^maxdepth || tree$count[tree$node==node]<minsplit ||
      tree$variance[tree$node==node]<minvar
  }

  split.eval <- function(av, sv, tv)
  {
    cond <- !is.na(av) & (if (is.numeric(av)) av<=as.numeric(sv) else av==sv)
    v1 <- tv[cond]
    n1 <- sum(cond)
    v0 <- tv[!cond]
    n0 <- sum(!cond)
    if (n1>0 && n0>0)
      weighted.dispersion(v1, v0)
    else
      Inf
  }

  split.select <- function(node)
  {
    splits <- data.frame()
    for (attribute in attributes)
    {
      uav <- sort(unique(data[nodemap==node,attribute]))
      if (length(uav)>1)
        splits <- rbind(splits,
                        data.frame(attribute=attribute,
                                   value=if (is.numeric(uav))
                                           midbrk(uav)
                                         else as.character(uav),
                                   stringsAsFactors=FALSE))
    }

    if (nrow(splits)>0)
      splits$eval <- sapply(1:nrow(splits),
                            function(s)
                            split.eval(data[nodemap==node,splits$attribute[s]],
                                       splits$value[s],
                                       data[nodemap==node,target]))
    if ((best.eval <- min(splits$eval))<Inf)
      tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
    best.eval
  }

  split.apply <- function(node)
  {
    tree <<- rbind(tree,
                   data.frame(node=(2*node):(2*node+1), attribute=NA, value=NA, target=NA,
                              count=NA, mean=NA, variance=NA))

    av <- data[[tree$attribute[tree$node==node]]]
    cond <- !is.na(av) & (if (is.numeric(av))
                            av<=as.numeric(tree$value[tree$node==node])
                          else av==tree$value[tree$node==node])
    nodemap[nodemap==node & cond] <<- 2*node
    nodemap[nodemap==node & !cond] <<- 2*node+1
  }

  tree <- nodemap <- node <- NULL
  target <- y.var(formula)
  attributes <- x.vars(formula, data)

  init()
  while (is.finite(node))
  {
    target.summary(node)
    target.value(node)
    if (!stop.criteria(node))
      if (split.select(node)<Inf)
        split.apply(node)
    node <- next.node(node)
  }
  `class<-`(tree, "regtree")
}

## convert a regtree object to a data frame
as.data.frame.regtree <- function(x, row.names=NULL, optional=FALSE, ...)
{ as.data.frame(unclass(x), row.names=row.names, optional=optional) }

  # grow a regression tree for the weatherr data
tree <- grow.regtree(playability~., weatherr)

  # data frame conversion
as.data.frame(tree)
##############
#9-5-1.R
##############
## regression tree prediction
predict.regtree <- function(tree, data)
{
  descend <- function(node)
  {
    if (!is.na(tree$attribute[tree$node==node]))  # unless reached a leaf
    {
      av <- data[[tree$attribute[tree$node==node]]]
      cond <- !is.na(av) & (if (is.numeric(av))
                              av<=as.numeric(tree$value[tree$node==node])
                            else av==tree$value[tree$node==node])
      nodemap[nodemap==node & cond] <<- 2*node
      nodemap[nodemap==node & !cond] <<- 2*node+1
      descend(2*node)
      descend(2*node+1)
    }
  }

  nodemap <- rep(1, nrow(data))
  descend(1)
  tree$target[match(nodemap, tree$node)]
}

  # regression tree prediction for the weatherr data
predict(tree, weatherr)
##############
#9-7-1.R
##############
## a simple regression tree growing implementation
grow.regtree.frac <- function(formula, data, minvar=0.005, minsplit=2, maxdepth=8)
{
  nmn <- function(node) { nodemap[,"node"]==node }            # nodemap entries for node node
  inn <- function(node)
  { nodemap[nodemap[,"node"]==node,"instance"] }                  # instances at node node
  wgn <- function(node) { nodemap[nodemap[,"node"]==node,"weight"] }   # weights at node node

  init <- function()
  {
    tree <<- data.frame(node=1, attribute=NA, value=NA, target=NA,
                        count=NA, mean=NA, variance=NA)
    nodemap <<- cbind(instance=1:nrow(data), node=rep(1, nrow(data)),
                      weight=rep(1, nrow(data)))
    node <<- 1
  }

  next.node <- function(node)
  {
    if (any(opn <- tree$node>node))
      min(tree$node[opn])
    else Inf
  }

  target.summary <- function(node)
  {
    tree$count[tree$node==node] <<- sum(wgn(node))
    tree$mean[tree$node==node] <<- weighted.mean(data[inn(node),target], wgn(node))
    tree$variance[tree$node==node] <<- weighted.var1(data[inn(node),target], wgn(node))
  }

  target.value <- function(node)
  {
    tree$target[tree$node==node] <<- tree$mean[tree$node==node]
  }

  stop.criteria <- function(node)
  {
    node>=2^maxdepth || tree$count[tree$node==node]<minsplit ||
      tree$variance[tree$node==node]<minvar
  }

  split.eval <- function(av, sv, tv, w)
  {
    cond <- if (is.numeric(av)) av<=as.numeric(sv) else av==sv
    cond1 <- !is.na(av) & cond   # true split outcome
    cond0 <- !is.na(av) & !cond  # false split outcome

    v1 <- tv[cond1]
    n1 <- sum(w[cond1])
    w1 <- w[cond1]
    v0 <- tv[cond0]
    n0 <- sum(w[cond0])
    w0 <- w[cond0]
    vm <- tv[is.na(av)]
    nm <- sum(w[is.na(av)])
    wm <- w[is.na(av)]

    if (nm>0)
    {
      p1 <- if (n1+n0>0) n1/(n1+n0) else 0.5
      p0 <- 1-p1
      v1 <- c(v1, vm)
      w1 <- c(w1, p1*wm)
      v0 <- c(v0, vm)
      w0 <- c(w0, p0*wm)
    }

    if (n1>0 && n0>0)
      weighted.dispersion(v1, v0, w1, w0, disp=weighted.var1)
    else
      Inf
  }

  split.select <- function(node)
  {
    splits <- data.frame()
    for (attribute in attributes)
    {
      uav <- sort(unique(data[inn(node),attribute]))
      if (length(uav)>1)
        splits <- rbind(splits,
                        data.frame(attribute=attribute,
                                   value=if (is.numeric(uav))
                                           midbrk(uav)
                                         else as.character(uav),
                                   stringsAsFactors=FALSE))
    }

    if (nrow(splits)>0)
      splits$eval <- sapply(1:nrow(splits),
                            function(s)
                            split.eval(data[inn(node),splits$attribute[s]],
                                       splits$value[s],
                                       data[inn(node),target], wgn(node)))
    if ((best.eval <- min(splits$eval))<Inf)
      tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
    best.eval
  }

  split.apply <- function(node)
  {
    tree <<- rbind(tree,
                   data.frame(node=(2*node):(2*node+1), attribute=NA, value=NA, target=NA,
                              count=NA, mean=NA, variance=NA))

    av <- data[nodemap[,"instance"],tree$attribute[tree$node==node]]
    cond <- if (is.numeric(av)) av<=as.numeric(tree$value[tree$node==node])
            else av==tree$value[tree$node==node]
    cond1 <- !is.na(av) & cond   # true split outcome
    cond0 <- !is.na(av) & !cond  # false split outcome

    n1 <- sum(nodemap[nmn(node) & cond1,"weight"])
    n0 <- sum(nodemap[nmn(node) & cond0,"weight"])
    nm <- sum(nodemap[nmn(node) & is.na(av),"weight"])

    nodemap[nmn(node) & cond1,"node"] <<- 2*node
    nodemap[nmn(node) & cond0,"node"] <<- 2*node+1

    if (nm>0)
    {
      p1 <- if (n1+n0>0) n1/(n1+n0) else 0.5
      p0 <- 1-p1
      newnn <- nodemap[nmn(node) & is.na(av),,drop=FALSE]
      nodemap[nmn(node) & is.na(av),"weight"] <<-
        p1*nodemap[nmn(node) & is.na(av),"weight"]
      nodemap[nmn(node) & is.na(av),"node"] <<- 2*node
      newnn[,"weight"] <- p0*newnn[,"weight"]
      newnn[,"node"] <- 2*node+1
      nodemap <<- rbind(nodemap, newnn)
    }
  }

  tree <- nodemap <- node <- NULL
  target <- y.var(formula)
  attributes <- x.vars(formula, data)

  init()
  while (is.finite(node))
  {
    target.summary(node)
    target.value(node)
    if (!stop.criteria(node))
      if (split.select(node)<Inf)
        split.apply(node)
    node <- next.node(node)
  }
  `class<-`(tree, "regtree.frac")
}

## convert a regtree.frac object to a data frame
as.data.frame.regtree.frac <- function(x, row.names=NULL, optional=FALSE, ...)
{ as.data.frame(unclass(x), row.names=row.names, optional=optional) }

  # grow a regression tree for the weatherr data with missing attribute values
weatherrm <- weatherr
weatherrm$outlook[1] <- NA
weatherrm$humidity[1:2] <- NA
treem <- grow.regtree.frac(playability~., weatherrm)

  # data frame conversion
as.data.frame(treem)
##############
#9-7-2.R
##############
## regression tree prediction
## with missing value support using fractional instances
predict.regtree.frac <- function(tree, data)
{
  nmn <- function(node) { nodemap[,"node"]==node }  # nodemap entries for node node

  descend <- function(node)
  {
    if (!is.na(tree$attribute[tree$node==node]))  # unless reached a leaf
    {
      av <- data[nodemap[,"instance"],tree$attribute[tree$node==node]]
      cond <- if (is.numeric(av)) av<=as.numeric(tree$value[tree$node==node])
              else av==tree$value[tree$node==node]
      cond1 <- !is.na(av) & cond   # true split outcome
      cond0 <- !is.na(av) & !cond  # false split outcome

      nodemap[nmn(node) & cond1, "node"] <<- 2*node
      nodemap[nmn(node) & cond0, "node"] <<- 2*node+1

      if (sum(nodemap[nmn(node) & is.na(av), "weight"])>0)
      {
        n1 <- tree$count[tree$node==2*node]
        n0 <- tree$count[tree$node==2*node+1]
        p1 <- if (n1+n0>0) n1/(n1+n0) else 0.5
        p0 <- 1-p1

        newnn <- nodemap[nmn(node) & is.na(av),,drop=FALSE]
        nodemap[nmn(node) & is.na(av),"weight"] <<-
          p1*nodemap[nmn(node) & is.na(av),"weight"]
        nodemap[nmn(node) & is.na(av), "node"] <<- 2*node
        newnn[,"weight"] <- p0*newnn[,"weight"]
        newnn[,"node"] <- 2*node+1
        nodemap <<- rbind(nodemap, newnn)
      }

      descend(2*node)
      descend(2*node+1)
    }
  }

  nodemap <- cbind(instance=1:nrow(data), node=rep(1, nrow(data)),
                   weight=rep(1, nrow(data)))
  descend(1)

  votes <- merge(nodemap, as.data.frame(tree)[,c("node", "target")])
  as.numeric(by(votes, votes$instance,
                function(v) weighted.mean(v$target, v$weight)))
}

  # regression tree prediction for the weatherr data with missing attribute values
predict(treem, weatherrm)
##############
#9-8-1.R
##############
## a simple model tree growing implementation
grow.modtree <- function(formula, data, minvar=0.005, minsplit=2, maxdepth=8)
{
  init <- function()
  {
    tree <<- data.frame(node=1, attribute=NA, value=NA,
                        count=NA, mean=NA, variance=NA)
    models <<- list()
    nodemap <<- rep(1, nrow(data))
    node <<- 1
  }

  next.node <- function(node)
  {
    if (any(opn <- tree$node>node))
      min(tree$node[opn])
    else Inf
  }

  target.summary <- function(node)
  {
    tree$count[tree$node==node] <<- sum(nodemap==node)
    tree$mean[tree$node==node] <<- mean(data[[target]][nodemap==node])
    tree$variance[tree$node==node] <<- var1(data[[target]][nodemap==node])
  }

  model <- function(node)
  {
    attrs <- drop1val(attributes, data[nodemap==node,])
    models <<- c(models,
                 list(lm(make.formula(target, if (length(attrs)==0) 1 else attrs),
                         data[nodemap==node,])))
  }

  stop.criteria <- function(node)
  {
    node>=2^maxdepth || tree$count[tree$node==node]<minsplit ||
      tree$variance[tree$node==node]<minvar
  }

  split.eval <- function(av, sv, tv)
  {
    cond <- !is.na(av) & (if (is.numeric(av)) av<=as.numeric(sv) else av==sv)
    v1 <- tv[cond]
    n1 <- sum(cond)
    v0 <- tv[!cond]
    n0 <- sum(!cond)
    if (n1>0 && n0>0)
      weighted.dispersion(v1, v0)
    else
      Inf
  }

  split.select <- function(node)
  {
    splits <- data.frame()
    for (attribute in attributes)
    {
      uav <- sort(unique(data[nodemap==node,attribute]))
      if (length(uav)>1)
        splits <- rbind(splits,
                        data.frame(attribute=attribute,
                                   value=if (is.numeric(uav))
                                           midbrk(uav)
                                         else as.character(uav),
                                   stringsAsFactors=FALSE))
    }

    if (nrow(splits)>0)
      splits$eval <- sapply(1:nrow(splits),
                            function(s)
                            split.eval(data[nodemap==node,splits$attribute[s]],
                                       splits$value[s],
                                       data[nodemap==node,target]))
    if ((best.eval <- min(splits$eval))<Inf)
      tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
    best.eval
  }

  split.apply <- function(node)
  {
    tree <<- rbind(tree,
                   data.frame(node=(2*node):(2*node+1), attribute=NA, value=NA,
                              count=NA, mean=NA, variance=NA))

    av <- data[[tree$attribute[tree$node==node]]]
    cond <- !is.na(av) & (if (is.numeric(av))
                            av<=as.numeric(tree$value[tree$node==node])
                          else av==tree$value[tree$node==node])
    nodemap[nodemap==node & cond] <<- 2*node
    nodemap[nodemap==node & !cond] <<- 2*node+1
  }

  tree <- models <- nodemap <- node <- NULL
  target <- y.var(formula)
  attributes <- x.vars(formula, data)

  init()
  while (is.finite(node))
  {
    target.summary(node)
    model(node)
    if (! stop.criteria(node))
      if (split.select(node)<Inf)
        split.apply(node)
    node <- next.node(node)
  }
  `class<-`(list(structure=tree, models=models), "modtree")
}

  # grow a model tree for the weatherr data
mtree <- grow.modtree(playability~., weatherr)
  # tree structure
mtree$structure
##############
#9-8-2.R
##############
## model tree prediction
predict.modtree <- function(tree, data, m=10)
{
  descend <- function(node)
  {
    predn <- predict(models[[which(tree$node==node)]], data[nodemap==node,])
    if (is.na(tree$attribute[tree$node==node]))  # reached a leaf
      pred[nodemap==node] <<- predn
    else
    {
      av <- data[[tree$attribute[tree$node==node]]]
      cond <- !is.na(av) & (if (is.numeric(av))
                              av<=as.numeric(tree$value[tree$node==node])
                            else av==tree$value[tree$node==node])
      nodemap[left <- nodemap==node & cond] <<- 2*node
      nodemap[right <- nodemap==node & !cond] <<- 2*node+1
      descend(2*node)
      descend(2*node+1)

      leftn <- match(which(left), which(left|right))
      rightn <- match(which(right), which(left|right))
      pred[left] <<- (tree$count[tree$node==2*node]*pred[left] + m*predn[leftn])/
                     (tree$count[tree$node==2*node]+m)
      pred[right] <<- (tree$count[tree$node==2*node+1]*pred[right] + m*predn[rightn])/
                      (tree$count[tree$node==2*node+1]+m)
     }
  }

  models <- tree$models
  tree <- tree$structure
  nodemap <- rep(1, nrow(data))
  pred <- rep(NA, nrow(data))
  descend(1)
  pred
}

  # model tree prediction for the weatherr data
predict(mtree, weatherr, m=2)
##############
#10-1-1.R
##############
library(dmr.claseval)
library(rpart)

data(BostonHousing, package="mlbench")

set.seed(12)
rbh <- runif(nrow(BostonHousing))
bh.train <- BostonHousing[rbh>=0.33,]
bh.test <- BostonHousing[rbh<0.33,]

bh.tree <- rpart(medv~., bh.train)
##############
#10-2-1.R
##############
res <- function(pred.y, true.y) { true.y-pred.y }

bh.res <- res(predict(bh.tree, bh.test), bh.test$medv)
summary(bh.res)
summary(abs(bh.res))

boxplot(bh.res, main="Residual boxplot")
hist(bh.res, main="Residual histogram")
plot(bh.test$medv, bh.res, main="Residual plot")
##############
#10-2-2.R
##############
mae <- function(pred.y, true.y) { mean(abs(true.y-pred.y)) }

mae(predict(bh.tree, bh.test), bh.test$medv)
##############
#10-2-3.R
##############
mse <- function(pred.y, true.y) { mean((true.y-pred.y)^2) }

mse(predict(bh.tree, bh.test), bh.test$medv)
##############
#10-2-4.R
##############
rmse <- function(pred.y, true.y) { sqrt(mse(pred.y, true.y)) }

rmse(predict(bh.tree, bh.test), bh.test$medv)
##############
#10-2-5.R
##############
rae <- function(pred.y, true.y)
{ mae(pred.y, true.y)/mean(abs(true.y-mean(true.y))) }

rae(predict(bh.tree, bh.test), bh.test$medv)
##############
#10-2-6.R
##############
r2 <- function(pred.y, true.y)
{ 1 - length(true.y)*mse(pred.y, true.y)/((length(true.y)-1)*var(true.y)) }

r2(predict(bh.tree, bh.test), bh.test$medv)
##############
#10-2-7.R
##############
cor(predict(bh.tree, bh.test), bh.test$medv, method="pearson")
cor(predict(bh.tree, bh.test), bh.test$medv, method="spearman")
##############
#10-2-8.R
##############
wmae <- function(pred.y, true.y, w=rep(1, length(true.y)))
{ weighted.mean(abs(true.y-pred.y), w) }

wmse <- function(pred.y, true.y, w=rep(1, length(true.y)))
{ weighted.mean((true.y-pred.y)^2, w) }

wrmse <- function(pred.y, true.y, w=rep(1, length(true.y)))
{ sqrt(wmse(pred.y, true.y, w)) }

wrae <- function(pred.y, true.y, w=rep(1, length(true.y)))
{ wmae(pred.y, true.y, w)/weighted.mean(abs(true.y-weighted.mean(true.y, w)), w) }

wr2 <- function(pred.y, true.y, w=rep(1, length(true.y)))
{
  1-weighted.mean((true.y-pred.y)^2, w)/
      weighted.mean((true.y-weighted.mean(true.y, w))^2, w)
}

  # double weight for medv>25
bh.wtest <- ifelse(bh.test$medv>25, 2, 1)

wmae(predict(bh.tree, bh.test), bh.test$medv, bh.wtest)
wmse(predict(bh.tree, bh.test), bh.test$medv, bh.wtest)
wrmse(predict(bh.tree, bh.test), bh.test$medv, bh.wtest)
wrae(predict(bh.tree, bh.test), bh.test$medv, bh.wtest)
wr2(predict(bh.tree, bh.test), bh.test$medv, bh.wtest)
##############
#10-2-9.R
##############
mls <- function(pred.y, true.y, loss) { mean(loss(pred.y, true.y)) }

loss.abs <- function(pred.y, true.y) { abs(true.y-pred.y) }

loss.square <- function(pred.y, true.y) { (true.y-pred.y)^2 }

loss.asymmetric <- function(loss, p=1, node=1)
{
  function(pred.y, true.y)
  {
    ifelse(res(pred.y, true.y)>0, p*loss(pred.y, true.y), node*loss(pred.y, true.y))
  }
}

mls(predict(bh.tree, bh.test), bh.test$medv, loss.abs)
mls(predict(bh.tree, bh.test), bh.test$medv, loss.square)
mls(predict(bh.tree, bh.test), bh.test$medv, loss.asymmetric(loss.abs, 2, 1))
##############
#10-3-1.R
##############
  # hold-out regression tree evaluation for the Boston Housing data
bh.ho <- holdout(rpart, medv~., BostonHousing, n=10)
mae(bh.ho$pred, bh.ho$true)
mse(bh.ho$pred, bh.ho$true)
rmse(bh.ho$pred, bh.ho$true)
rae(bh.ho$pred, bh.ho$true)
r2(bh.ho$pred, bh.ho$true)
cor(bh.ho$pred, bh.ho$true, method="pearson")
cor(bh.ho$pred, bh.ho$true, method="spearman")
##############
#10-3-2.R
##############
  # regression tree cross-validation for the BostonHousing data
bh.cv3 <- crossval(rpart, medv~., BostonHousing, k=3)
mse(bh.cv3$pred, bh.cv3$true)
bh.cv5 <- crossval(rpart, medv~., BostonHousing, k=5)
mse(bh.cv5$pred, bh.cv5$true)
bh.cv10 <- crossval(rpart, medv~., BostonHousing, k=10)
mse(bh.cv10$pred, bh.cv10$true)
bh.cv20 <- crossval(rpart, medv~., BostonHousing, k=20)
mse(bh.cv20$pred, bh.cv20$true)
##############
#10-3-3.R
##############
  # leave-one-out regression tree evaluation for the BostonHousing data
bh.l1o <- leave1out(rpart, medv~., BostonHousing)
mse(bh.l1o$pred, bh.l1o$true)
##############
#10-3-4.R
##############
  # 20x bootstrap regression tree evaluation for the BostonHousing data
bh.bs20 <- bootstrap(rpart, medv~., BostonHousing, w=1, m=20)
mse(bh.bs20$pred, bh.bs20$true)

  # 20x .632 bootstrap regression tree evaluation for the BostonHousing data
bh.632bs20 <- bootstrap(rpart, medv~., BostonHousing, m=20)
wmse(bh.632bs20$pred, bh.632bs20$true, bh.632bs20$w)
##############
#10-3-5.R
##############
  # the commented line runs a 200-repetition experiment, which takes a long time
#bh.ebv <- eval.bias.var(rpart, medv~., BostonHousing, perf=mse, wperf=wmse, n=200)
  # this can be used for a quick illustration
bh.ebv <- eval.bias.var(rpart, medv~., BostonHousing, perf=mse, wperf=wmse, n=10)

boxplot(bh.ebv$performance[,-1], main="Error", las=2)
lines(c(0, 13), rep(mean(bh.ebv$performance[,1]), 2), lty=2)
barplot(bh.ebv$bias, main="Bias", las=2)
barplot(bh.ebv$variance, main="Variance", las=2)
##############
#11-1-1.R
##############
library(dmr.util)
library(dmr.trans)

data(weathercl, package="dmr.data")
##############
#11-2-1.R
##############
dissmat <- function(data, diss)
{
  as.dist(outer(1:nrow(data), 1:nrow(data),
                Vectorize(function(i, j)
                          if (j<=i) diss(data[i,], data[j,]) else NA)),
          diag=TRUE, upper=TRUE)
}

  # dummy dissimilarity matrix for the weathercl data
dummy.diss <- function(x1, x2)
{ abs(as.integer(row.names(x1))-as.integer(row.names(x2))) }

dissmat(weathercl, dummy.diss)
##############
#11-3-1.R
##############
avdiff <- function(x1, x2)
{
  mapply(function(v1, v2) ifelse(is.numeric(v1), v1-v2, v1!=v2), x1, x2)
}

euc.dist <- function(x1, x2) { sqrt(sum(avdiff(x1,x2)^2, na.rm=TRUE)) }

  # Euclidean distance dissimilarity matrix for the weathercl data
dissmat(weathercl, euc.dist)
##############
#11-3-2.R
##############
mink.dist <- function(x1, x2, p) { (sum(abs(avdiff(x1,x2))^p, na.rm=TRUE))^(1/p) }

  # Minkowski distance dissimilarity matrices for the weathercl data
dissmat(weathercl, function (x1, x2) mink.dist(x1, x2, 1))
dissmat(weathercl, function (x1, x2) mink.dist(x1, x2, 3))
##############
#11-3-3.R
##############
man.dist <- function(x1, x2) { mink.dist(x1, x2, 1) }

  # Manhattan distance dissimilarity matrix for the weathercl data
dissmat(weathercl, function (x1, x2) man.dist(x1, x2))
##############
#11-3-4.R
##############
ravdiff <- function(x1, x2)
{
  mapply(function(v1, v2) ifelse(is.numeric(v1), (v1-v2)/(abs(v1)+abs(v2)), v1!=v2),
         x1, x2)
}

can.dist <- function(x1, x2) { sum(abs(ravdiff(x1,x2)), na.rm=TRUE) }

  # Canberra distance dissimilarity matrix for the weathercl data
dissmat(weathercl, function (x1, x2) can.dist(x1, x2))
##############
#11-3-5.R
##############
cheb.dist <- function(x1, x2) { max(abs(avdiff(x1,x2)), na.rm=TRUE) }

  # Chebyshev distance dissimilarity matrix for the weathercl
dissmat(weathercl, cheb.dist)
  # roughly the same as
dissmat(weathercl, function (x1, x2) mink.dist(x1, x2, 10))
##############
#11-3-6.R
##############
ham.dist <- function(x1, x2) { sum(x1!=x2, na.rm=TRUE) }

  # Hamming distance dissimilarity matrix for the weathercl
dissmat(weathercl, ham.dist)
##############
#11-3-7.R
##############
gower.coef <- function(x1, x2, rngs)
{
  mean(mapply(function(v1, v2, r) ifelse(is.numeric(v1), abs(v1-v2)/r, v1!=v2),
              x1, x2, rngs), na.rm=TRUE)
}

  # Gower's coefficient dissimilarity matrix for the weathercl data
dissmat(weathercl, function (x1, x2) gower.coef(x1, x2, ranges(weathercl)))
##############
#11-3-8.R
##############
weathercl.std <- predict.std(std.all(.~., weathercl), weathercl)
dissmat(weathercl.std, euc.dist)
##############
#11-4-1.R
##############
discode(~., weathercl[1,])
discode(~., weathercl[5,])
##############
#11-4-2.R
##############
pearson.sim <- function(x1, x2)
{
  cor(unlist(discode(~., x1)), unlist(discode(~., x2)), method="pearson",
      use="pairwise.complete.obs")
}

  # Pearson similarity matrix for the weathercl data
dissmat(weathercl, pearson.sim )
##############
#11-4-3.R
##############
spearman.sim <- function(x1, x2)
{
  cor(unlist(discode(~., x1)), unlist(discode(~., x2)), method="spearman",
      use="pairwise.complete.obs")
}

  # Spearman similarity matrix for the weathercl data
dissmat(weathercl, spearman.sim )
##############
#11-4-4.R
##############
cos.sim <- function(x1, x2) { cosine(discode(~., x1), discode(~., x2)) }

  # cosine similarity matrix for the weathercl data
dissmat(weathercl, cos.sim)
##############
#12-1-1.R
##############
library(dmr.claseval)
library(dmr.dissim)
library(dmr.stats)
library(dmr.trans)

library(rpart)
library(rpart.plot)

data(weathercl, package="dmr.data")
data(iris)
data(Glass, package="mlbench")

set.seed(12)

ri <- runif(nrow(iris))
i.train <- iris[ri>=0.33,]
i.test <- iris[ri<0.33,]

rg <- runif(nrow(Glass))
g.train <- Glass[rg>=0.33,]
g.test <- Glass[rg<0.33,]

wcl.std <- predict.std(std.all(.~., weathercl), weathercl)

i.stdm <- std.all(Species~., i.train)
i.std.train <- predict.std(i.stdm, i.train)
i.std.test <- predict.std(i.stdm, i.test)

g.stdm <- std.all(Type~., g.train)
g.std.train <- predict.std(g.stdm, g.train)
g.std.test <- predict.std(g.stdm, g.test)
##############
#12-2-1.R
##############
k.centers <- function(data, k, diss=euc.dist, max.iter=10,
                      init=k.centers.init.rand,
                      assign=k.centers.assign,
                      adjust=k.centers.adjust.dummy)
{
  dm <- dissmat(data, diss)
  centers <- init(data, k)
  clustering <- NULL
  iter <- 0
  repeat
  {
    iter <- iter+1
    clustering.old <- clustering
    clustering <- assign(centers, data, diss, dm)
    centers <- adjust(clustering, data, k, diss, dm)
    if (iter >= max.iter || all(clustering==clustering.old))
      break
  }

  `class<-`(list(centers=centers, clustering=clustering), "k.centers")
}

k.centers.init.rand <- function(data, k) { data[sample(1:nrow(data), k),] }

k.centers.assign <- function(centers, data, diss, dm)
{
  center.diss <- function(i)
  { sapply(1:nrow(centers), function(d) diss(data[i,], centers[d,])) }
  assign1 <- function(i) { which.min(center.diss(i)) }
  sapply(1:nrow(data), assign1)
}

predict.k.centers <- function(model, data, diss=euc.dist)
{
  k.centers.assign(model$centers, data, diss)
}

k.centers.adjust.dummy <- function(clustering, data, k, diss, dm)
{
  do.call(rbind, lapply(1:k, function(d) data[which.max(clustering==d),]))
}

  # dummy k-centers clustering for the weathercl data
k.centers(wcl.std, 3)
##############
#12-3-1.R
##############
## attribute value means, medians, or modes
attr.mm <- function(data, mc=mean, md=modal)
{
  data.frame(`names<-`(lapply(data, function(v)
                              if (is.numeric(v)) mc(v) else md(v)),
                       names(data)))
}

## k-means center adjustment
k.centers.adjust.mean <- function(clustering, data, k, diss, dm)
{
  do.call(rbind, lapply(1:k, function(d) attr.mm(data[clustering==d,])))
}

  # k-means clustering
w.kmeans <- k.centers(wcl.std, 3, adjust=k.centers.adjust.mean)
w.kmeans$centers

i.kmeans <- k.centers(i.std.train[,-5], 3, adjust=k.centers.adjust.mean)
g.kmeans <- k.centers(g.std.train[,-10], 7, adjust=k.centers.adjust.mean)

  # k-means prediction
w.kmeans$clustering
predict(w.kmeans, wcl.std)

i.pred.kmeans <- predict(i.kmeans, i.std.test[,-5])
g.pred.kmeans <- predict(g.kmeans, g.std.test[,-10])

  # clusters vs. classes on the training set
table(i.kmeans$clustering, i.std.train$Species)
table(g.kmeans$clustering, g.std.train$Type)

  # clusters vs. classes on the test set
table(predict(i.kmeans, i.std.test[,-5]), i.std.test$Species)
table(predict(g.kmeans, g.std.test[,-10]), g.std.test$Type)

  # attribute distribution within clusters for the Iris data
par(mfrow=c(2, 2))
for (attr in names(i.std.train)[1:4])
  boxplot(i.std.train[[attr]]~i.kmeans$clustering, main=attr)
##############
#12-4-1.R
##############
## k-medians center adjustment
k.centers.adjust.median <- function(clustering, data, k, diss, dm)
{
  do.call(rbind, lapply(1:k, function(d) attr.mm(data[clustering==d,], mc=median)))
}

  # k-medians clustering
w.kmedians <- k.centers(wcl.std, 3, adjust=k.centers.adjust.median)
w.kmedians$centers

i.kmedians <- k.centers(i.std.train[,-5], 3, adjust=k.centers.adjust.median)
g.kmedians <- k.centers(g.std.train[,-10], 7, adjust=k.centers.adjust.median)

  # k-medians prediction
w.kmedians$clustering
predict(w.kmedians, wcl.std)

i.pred.kmedians <- predict(i.kmedians, i.std.test[,-5])
g.pred.kmedians <- predict(g.kmedians, g.std.test[,-10])

  # clusters vs. classes on the training set
table(i.kmedians$clustering, i.std.train$Species)
table(g.kmedians$clustering, g.std.train$Type)

  # clusters vs. classes on the test set
table(predict(i.kmedians, i.std.test[,-5]), i.std.test$Species)
table(predict(g.kmedians, g.std.test[,-10]), g.std.test$Type)

  # attribute distribution within clusters for the Iris data
par(mfrow=c(2, 2))
for (attr in names(i.std.train)[1:4])
  boxplot(i.std.train[[attr]]~i.kmedoids$clustering, main=attr)
##############
#12-4-2.R
##############
## medoid for data with respect to dissimilarity matrix dm
medoid <- function(data, dm)
{
  data[which.min(colMeans(dm)),]
}

## k-medoids center adjustment
k.centers.adjust.medoid <- function(clustering, data, k, diss, dm)
{
  do.call(rbind, lapply(1:k, function(d)
                             medoid(data[clustering==d,],
                                    as.matrix(dm)[clustering==d, clustering==d])))
}

  # k-medoids clustering
w.kmedoids <- k.centers(wcl.std, 3, adjust=k.centers.adjust.medoid)
w.kmedoids$centers

i.kmedoids <- k.centers(i.std.train[,-5], 3, adjust=k.centers.adjust.medoid)
g.kmedoids <- k.centers(g.std.train[,-10], 7, adjust=k.centers.adjust.medoid)

  # k-medoids prediction
w.kmedoids$clustering
predict(w.kmedoids, wcl.std)

i.pred.kmedoids <- predict(i.kmedoids, i.std.test[,-5])
g.pred.kmedoids <- predict(g.kmedoids, g.std.test[,-10])

  # clusters vs. classes on the training set
table(i.kmedoids$clustering, i.std.train$Species)
table(g.kmedoids$clustering, g.std.train$Type)

  # clusters vs. classes on the test set
table(predict(i.kmedoids, i.std.test[,-5]), i.std.test$Species)
table(predict(g.kmedoids, g.std.test[,-10]), g.std.test$Type)

  # attribute distribution within clusters for the Iris data
par(mfrow=c(2, 2))
for (attr in names(i.std.train)[1:4])
  boxplot(i.std.train[[attr]]~i.kmedoids$clustering, main=attr)
##############
#12-6-1.R
##############
  # explicit decision tree representations of k-means models
i.kmeans.tree <- rpart(cluster~.,
                       cbind(i.train, cluster=as.factor(i.kmeans$clustering)),
                       minsplit=min(table(i.kmeans$clustering)), cp=0.05)
g.kmeans.tree <- rpart(cluster~.,
                       cbind(g.train, cluster=as.factor(g.kmeans$clustering)),
                       minsplit=min(table(g.kmeans$clustering)), cp=0.05)

  # cluster membership prediction tree plots
prp(i.kmeans.tree, varlen=0, faclen=0, main="Iris")
prp(g.kmeans.tree, varlen=0, faclen=0, main="Glass")

  # predicted vs. true clusters
confmat(predict(i.kmeans.tree, i.train, type="c"), i.kmeans$clustering)
confmat(predict(i.kmeans.tree, i.test, type="c"),
        predict(i.kmeans, i.std.test[,-5], euc.dist))

confmat(predict(g.kmeans.tree, g.train, type="c"), g.kmeans$clustering)
confmat(predict(g.kmeans.tree, g.test, type="c"),
        predict(g.kmeans, g.std.test[,-10], euc.dist))
##############
#13-1-1.R
##############
library(dmr.dissim)
library(dmr.trans)

library(cluster)

data(weathercl, package="dmr.data")
data(iris)
data(Glass, package="mlbench")

set.seed(12)

ri <- runif(nrow(iris))
i.train <- iris[ri>=0.33,]
i.test <- iris[ri<0.33,]

rg <- runif(nrow(Glass))
g.train <- Glass[rg>=0.33,]
g.test <- Glass[rg<0.33,]

wcl.std <- predict.std(std.all(.~., weathercl), weathercl)

i.stdm <- std.all(Species~., i.train)
i.std.train <- predict.std(i.stdm, i.train)
i.std.test <- predict.std(i.stdm, i.test)

g.stdm <- std.all(Type~., g.train)
g.std.train <- predict.std(g.stdm, g.train)
g.std.test <- predict.std(g.stdm, g.test)
##############
#13-2-1.R
##############
dg.l4 <- lapply(1:8, function(i)
                     {
                       d <- list(i)
                       attr(d, "members") <- 1
                       attr(d, "height") <- 0
                       attr(d, "leaf") <- TRUE
                       attr(d, "label") <- i
                       `class<-`(d, "dendrogram")
                     })

dgmerge <- function(dg, i1, i2)
{
  d <- list(dg[[i1]], dg[[i2]])
  attr(d, "members") <- attr(d[[1]], "members")+attr(d[[2]], "members")
  attr(d, "height") <- 1+max(attr(d[[1]], "height"), attr(d[[2]], "height"))
  attr(d, "leaf") <- FALSE
  lab <- if (is.null(attr(d[[1]], "edgetext"))) "label" else "edgetext"
  attr(d, "edgetext") <- paste(attr(d[[1]], lab), attr(d[[2]], lab), sep="+")
  `class<-`(d, "dendrogram")
}

dg.l3 <- lapply(seq(1, length(dg.l4)-1, 2), function(i) dgmerge(dg.l4, i, i+1))
dg.l2 <- lapply(seq(1, length(dg.l3)-1, 2), function(i) dgmerge(dg.l3, i, i+1))
dg.l1 <- lapply(seq(1, length(dg.l2)-1, 2), function(i) dgmerge(dg.l2, i, i+1))

plot(dg.l1[[1]], center=TRUE)
##############
#13-3-1.R
##############
## agglomerative hierarchical clustering
ahc <- function(data, linkf=ahc.size, diss=euc.dist, bottom=1:nrow(data))
{
    # hclust-compatible cluster id scheme
  clid <- function(d)
  { if (d>length(bottom.clusters)) d-length(bottom.clusters) else -d }

  dm <- as.matrix(dissmat(data, diss))  # instance dissimilarity matrix for linkage

  bottom.clusters <- unique(bottom)  # bottom-level clusters
  clustering <- bottom               # current cluster assignment
  clusters <- bottom.clusters        # current set of clusters

  merge <- NULL   # merge matrix
  height <- NULL  # height vector
  order <- NULL   # order vector

  links <- outer(1:length(clusters), 1:length(clusters),
                 Vectorize(function(i1, i2)
                           if (i1<i2)
                             linkf(clustering, clusters[i1], clusters[i2], data,
                                   diss, dm)
                           else NA))
  while(length(clusters)>1)
  {
    mli <- arrayInd(which.min(links), dim(links))  # minimum link index
    d1 <- clusters[i1 <- mli[1]]
    d2 <- clusters[i2 <- mli[2]]
    d12 <- max(clusters)+1
        # merge d1 and d2 into d12
    merge <- rbind(merge, c(clid(d1), clid(d2)))
    height <- c(height, if (is.null(height) || links[i1,i2]>height[length(height)])
                          links[i1,i2]
                        else height[length(height)]+height[1])  # height correction
    clustering[clustering==d1 | clustering==d2] <- d12
    clusters <- clusters[-c(i1, i2)]
    links <- links[-c(i1, i2),,drop=FALSE]  # remove links for d1
    links <- links[,-c(i1, i2),drop=FALSE]  # remove links for d2
    if (length(clusters)>0)
    {
      links <- cbind(links, sapply(clusters,
                                   function(d) linkf(clustering, d, d12, data,
                                                     diss, dm)))
      links <- rbind(links, NA)  # keep the matrix square
    }
    clusters <- c(clusters, d12)
  }

  `class<-`(list(clustering=bottom, link=linkf, data=data,
                 merge=merge, height=height, order=-t(merge)[t(merge)<0]),
            "hcl")
}

## convert to hclust
as.hclust.hcl <- function(model) { `class<-`(unclass(model), c("hclust")) }

## size linkage (dummy)
ahc.size <- function(clustering, d1, d2, data, diss, dm)
{ sum(clustering==d1) + sum(clustering==d2) }

  # agglomerative clustering for the weathercl data
wcl.ahc.d <- ahc(wcl.std, linkf=ahc.size)
as.hclust(wcl.ahc.d)
##############
#13-3-2.R
##############
## single linkage
ahc.single <- function(clustering, d1, d2, data, diss, dm)
{ min(dm[clustering==d1, clustering==d2]) }

  # agglomerative hierarchical single-linkage clustering for the weathercl data
wcl.ahc.sl <- ahc(wcl.std, linkf=ahc.single)

  # agglomerative hierarchical single-linkage clustering for the iris data
i.ahc.sl <- ahc(i.std.train[,-5], linkf=ahc.single)

  # agglomerative hierarchical single-linkage clustering for the Glass data
g.ahc.sl <- ahc(g.std.train[,-10], linkf=ahc.single)
##############
#13-3-3.R
##############
## complete linkage
ahc.complete <- function(clustering, d1, d2, data, diss, dm)
{ max(dm[clustering==d1, clustering==d2]) }

  # agglomerative hierarchical complete-linkage clustering for the weathercl data
wcl.ahc.cl <- ahc(wcl.std, linkf=ahc.complete)

  # agglomerative hierarchical complete-linkage clustering for the iris data
i.ahc.cl <- ahc(i.std.train[,-5], linkf=ahc.complete)

  # agglomerative hierarchical complete-linkage clustering for the Glass data
g.ahc.cl <- ahc(g.std.train[,-10], linkf=ahc.complete)
##############
#13-3-4.R
##############
## average linkage
ahc.average <- function(clustering, d1, d2, data, diss, dm)
{ mean(dm[clustering==d1, clustering==d2]) }

  # agglomerative hierarchical average-linkage clustering for the weathercl data
wcl.ahc.al <- ahc(wcl.std, linkf=ahc.average)

  # agglomerative hierarchical average-linkage clustering for the iris data
i.ahc.al <- ahc(i.std.train[,-5], linkf=ahc.average)

  # agglomerative hierarchical average-linkage clustering for the Glass data
g.ahc.al <- ahc(g.std.train[,-10], linkf=ahc.average)
##############
#13-3-5.R
##############
## center (mean/mode) linkage
ahc.center <- function(clustering, d1, d2, data, diss, dm)
{ diss(attr.mm(data[clustering==d1,]), attr.mm(data[clustering==d2,])) }

  # agglomerative hierarchical center-linkage clustering for the weathercl data
wcl.ahc.ml <- ahc(wcl.std, linkf=ahc.center)

  # agglomerative hierarchical center-linkage clustering for the iris data
i.ahc.ml <- ahc(i.std.train[,-5], linkf=ahc.center)

  # agglomerative hierarchical center-linkage clustering for the Glass data
g.ahc.ml <- ahc(g.std.train[,-10], linkf=ahc.center)
##############
#13-3-6.R
##############
## Ward linkage
ahc.ward <- function(clustering, d1, d2, data, diss, dm)
{
  c1 <- attr.mm(data[clustering==d1,])
  c2 <- attr.mm(data[clustering==d2,])
  c12 <- attr.mm(data[clustering==d1 | clustering==d2,])

  sum(sapply(which(clustering==d1 | clustering==d2),
             function(i) diss(data[i,], c12)^2)) -
    sum(sapply(which(clustering==d1), function(i) diss(data[i,], c1)^2)) -
    sum(sapply(which(clustering==d2), function(i) diss(data[i,], c2)^2))
}

  # agglomerative hierarchical Ward-linkage clustering for the weathercl data
wcl.ahc.wl <- ahc(wcl.std, linkf=ahc.ward)

  # agglomerative hierarchical Ward-linkage clustering for the iris data
i.ahc.wl <- ahc(i.std.train[,-5], linkf=ahc.ward)

  # agglomerative hierarchical Ward-linkage clustering for the Glass data
g.ahc.wl <- ahc(g.std.train[,-10], linkf=ahc.ward)
##############
#13-4-1.R
##############
## divisive clustering using alg, which is assumed to be called:
##   alg(data, 2, ...)
dhc <- function(data, alg=pam, cls="clustering", cnt="medoids", centf=as.numeric,
                maxdepth=16, ...)
{
  clustering <- rep(1, nrow(data))  # cluster membership assignment
  centers <- NULL  # cluster centers
  merge <- NULL
  height <- NULL

  while (any(clustering>0))
  {
    d <- min(clustering[clustering>0])  # cluster to process
    if ((m <- sum(clustering==d))>1 && d<2^maxdepth)
    {

      cls.d <- if (m>2) (mod.d <- alg(data[clustering==d,], 2, ...))[[cls]] else 1:2
      centers <- c(list(if (m>2) mod.d[[cnt]]
                        else sapply(data[clustering==d,], centf)), centers)
      clustering[clustering==d] <- 2*d + (cls.d-1)
      merge <- rbind(c(2*d, 2*d+1), merge)
      height <- c(height, length(height)+1)
    }
    else
    {
      clustering[clustering==d] <- -d   # mark as leaf
      merge[merge==d] <- -d
    }
  }

  bottom <- unique(clustering)
  clustering <- (1:length(bottom))[match(clustering, bottom)]  # re-assign ids
  merge[merge<0] <- -(1:length(bottom))[match(merge[merge<0], bottom)]
  merge[merge>0][order(merge[merge>0])] <- sum(merge>0):1
  `class<-`(list(clustering=clustering, centers=centers, merge=merge, height=height,
                 order=-t(merge)[t(merge)<0]),
            "hcl")
}

  # divisive clustering for the weathercl data
wcl.dhc <- dhc(wcl.std)

  # divisive hierarchical clustering for the iris data
i.dhc <- dhc(i.std.train[,-5])
i.dhc.km <- dhc(i.std.train[,-5], alg=kmeans, cls="cluster", cnt="centers")
i.dhc.d3 <- dhc(i.std.train[,-5], maxdepth=3)
i.dhc.km.d3 <- dhc(i.std.train[,-5], alg=kmeans, cls="cluster", cnt="centers",
                   maxdepth=3)

  # divisive hierarchical clustering for the Glass data
g.dhc <- dhc(g.std.train[,-10]  )
g.dhc.km <- dhc(g.std.train[,-10], alg=kmeans, cls="cluster", cnt="centers")
g.dhc.d3 <- dhc(g.std.train[,-10], maxdepth=3)
g.dhc.km.d3 <- dhc(g.std.train[,-10], alg=kmeans, cls="cluster", cnt="centers",
                   maxdepth=3)
##############
#13-5-1.R
##############
## convert to dendrogram
as.dendrogram.hcl <- function(model) { as.dendrogram(as.hclust(model)) }

## plot a hierarchical clustering dendrogram
plot.hcl <- function(model, ...)
{
  plot(as.dendrogram(model), center=TRUE, ...)
}

  # dendrogram plots for the weathercl data
par(mfrow=c(3, 2))
plot(wcl.ahc.sl, main="Single linkage")
plot(wcl.ahc.cl, main="Complete linkage")
plot(wcl.ahc.al, main="Average linkage")
plot(wcl.ahc.ml, main="Center linkage")
plot(wcl.ahc.wl, main="Ward linkage")
plot(wcl.dhc, main="Divisive clustering")
##############
#13-6-1.R
##############
## cut a hierarchical clustering model to k clusters
cut.hcl <- function(model, k)
{
  nc <- maxc <- nrow(model$merge)+1  # number of clusters
  k <- clip.val(k, 2, nc)            # make sure k is in the valid range

  clustering <- model$clustering
  merge <- model$merge
  height <- model$height
  while (nc>k)
  {
    mr <- merge[1,]    # merge to remove
    merge <- merge[-1,]
    clustering[clustering %in% -mr] <- (maxc <- maxc + 1)  # id for new leaf
    merge[merge>0] <- merge[merge>0]-1  # shift node numbers
    merge[merge==0] <- -maxc
    height <- height[-1]-(height[2]-height[1])
    nc <- nc-1
  }

  bottom <- unique(clustering)
  clustering <- (1:length(bottom))[match(clustering, bottom)]  # re-assign ids
  merge[merge<0] <- -(1:length(bottom))[match(merge[merge<0], -bottom)]

  model$clustering <- clustering
  model$centers <- model$centers[-(1:(length(model$centers)-k+1))]
  model$merge <- merge
  model$height <- height
  model$order <- -t(merge)[t(merge)<0]
  model
}

  # cutting hierarchical clustering trees for the weathercl data
wcl.ahc.sl.c4 <- cut(wcl.ahc.sl, 4)
wcl.ahc.cl.c4 <- cut(wcl.ahc.cl, 4)
wcl.dhc.c4 <- cut(wcl.dhc, 4)

  # cutting hierarchical clustering trees for the iris data
i.ahc.sl.cd3 <- cut(i.ahc.sl, max(i.dhc.d3$clustering))
i.ahc.cl.cd3 <- cut(i.ahc.cl, max(i.dhc.d3$clustering))
i.ahc.al.cd3 <- cut(i.ahc.al, max(i.dhc.d3$clustering))
i.ahc.ml.cd3 <- cut(i.ahc.ml, max(i.dhc.d3$clustering))
i.ahc.wl.cd3 <- cut(i.ahc.wl, max(i.dhc.d3$clustering))
i.dhc.cd3 <- cut(i.dhc, max(i.dhc.d3$clustering))
  # verify i.dhc.cd3 and i.dhc.d3 are the same
all(i.dhc.cd3$clustering==i.dhc.d3$clustering)
all(i.dhc.cd3$merge==i.dhc.d3$merge)
all(sapply(1:length(i.dhc.cd3$centers),
           function(d) all(i.dhc.cd3$centers[[d]]==i.dhc.d3$centers[[d]])))

  # cutting hierarchical clustering trees for the Glass data
g.ahc.sl.cd3 <- cut(g.ahc.sl, max(g.dhc.d3$clustering))
g.ahc.cl.cd3 <- cut(g.ahc.cl, max(g.dhc.d3$clustering))
g.ahc.al.cd3 <- cut(g.ahc.al, max(g.dhc.d3$clustering))
g.ahc.ml.cd3 <- cut(g.ahc.ml, max(g.dhc.d3$clustering))
g.ahc.wl.cd3 <- cut(g.ahc.wl, max(g.dhc.d3$clustering))
g.dhc.cd3 <- cut(g.dhc, max(g.dhc.d3$clustering))
  # verify g.dhc.cd3 and g.dhc.d3 are the same
all(g.dhc.cd3$clustering==g.dhc.d3$clustering)
all(g.dhc.cd3$merge==g.dhc.d3$merge)
all(sapply(1:length(g.dhc.cd3$centers),
           function(d) all(g.dhc.cd3$centers[[d]]==g.dhc.d3$centers[[d]])))
##############
#13-6-2.R
##############
## hierarchical clustering prediction
predict.hcl <- function(model, data, ...)
{
  if (!is.null(model$data) && !is.null(model$link))
    predict.ahc(model, data, ...)
  else if (!is.null(model$centers))
    predict.dhc(model, data, ...)
}

## agglomerative hierarchical clustering prediction
predict.ahc <- function(model, data, diss=euc.dist)
{
  ext.data <- rbind(model$data, data)
  dm <- as.matrix(dissmat(ext.data, diss))  # dissimilarity matrix for linkage

  clusters <- sort(unique(model$clustering))
  x.clusters <- length(clusters) + 1:nrow(data)
  ext.clustering <- c(model$clustering, x.clusters)

  links <- outer(clusters, x.clusters,
                 Vectorize(function(d1, d2)
                           model$link(ext.clustering, d1, d2, ext.data, diss, dm)))
  apply(links, 2, which.min)
}

## divisive hierarchical clustering prediction
predict.dhc <- function(model, data, diss=euc.dist)
{
  centers <- do.call(rbind,
                     lapply(1:nrow(model$merge),
                            function(i)
                            model$centers[[i]][model$merge[i,]<0,]))
  clusters <- -t(model$merge)[t(model$merge)<0]
  centers <- centers[match(1:length(clusters), clusters),]  # reorder centers
  k.centers.assign(centers, data, diss)
}

  # hierarchical clustering prediction for the weathercl data
predict(wcl.ahc.cl, wcl.std)
predict(wcl.dhc, wcl.std)

  # hierarchical clustering prediction for the iris data
i.ahc.cl.cd3.pred <- predict(i.ahc.cl.cd3, i.std.test[,-5])
i.ahc.sl.cd3.pred <- predict(i.ahc.sl.cd3, i.std.test[,-5])
i.ahc.al.cd3.pred <- predict(i.ahc.al.cd3, i.std.test[,-5])
i.ahc.ml.cd3.pred <- predict(i.ahc.ml.cd3, i.std.test[,-5])
i.ahc.wl.cd3.pred <- predict(i.ahc.wl.cd3, i.std.test[,-5])
i.dhc.cd3.pred <- predict(i.dhc.cd3, i.std.test[,-5])

  # hierarchical clustering prediction for the Glass data
g.ahc.cl.cd3.pred <- predict(g.ahc.cl.cd3, g.std.test[,-10])
g.ahc.sl.cd3.pred <- predict(g.ahc.sl.cd3, g.std.test[,-10])
g.ahc.al.cd3.pred <- predict(g.ahc.al.cd3, g.std.test[,-10])
g.ahc.ml.cd3.pred <- predict(g.ahc.ml.cd3, g.std.test[,-10])
g.ahc.wl.cd3.pred <- predict(g.ahc.wl.cd3, g.std.test[,-10])
g.dhc.cd3.pred <- predict(g.dhc.cd3, g.std.test[,-10])
##############
#14-1-1.R
##############
library(dmr.claseval)
library(dmr.stats)
library(dmr.trans)
library(dmr.util)

library(cluster)

data(iris)

set.seed(12)
ri <- runif(nrow(iris))
i.train <- iris[ri>=0.33,]
i.test <- iris[ri<0.33,]

i.stdm <- std.all(Species~., i.train)
i.std.train <- predict.std(i.stdm, i.train)
i.std.test <- predict.std(i.stdm, i.test)

i.pam2.euc <- pam(i.std.train[,-5], 2, metric="euclidean")
i.pam3.euc <- pam(i.std.train[,-5], 3, metric="euclidean")
i.pam5.euc <- pam(i.std.train[,-5], 5, metric="euclidean")
i.pam7.euc <- pam(i.std.train[,-5], 7, metric="euclidean")

i.pam2.man <- pam(i.std.train[,-5], 2, metric="manhattan")
i.pam3.man <- pam(i.std.train[,-5], 3, metric="manhattan")
i.pam5.man <- pam(i.std.train[,-5], 5, metric="manhattan")
i.pam7.man <- pam(i.std.train[,-5], 7, metric="manhattan")
##############
#14-1-2.R
##############
## prediction for pam clustering models (only if created with stand=FALSE)
## using daisy or dist (selected via the dmf argument) for dissimilarity calculation
predict.pam <- function(model, data, dmf=daisy, ...)
{
  k.centers.assign(model$medoids, data,
                   function(x1, x2) dmf(rbind(x1, x2), ...))
}

## the same is applicable to clara clustering models
predict.clara <- predict.pam

  # test set predictions
i.pam2.euc.pred <- predict(i.pam2.euc, i.std.test[,-5])
i.pam3.euc.pred <- predict(i.pam3.euc, i.std.test[,-5])
i.pam5.euc.pred <- predict(i.pam5.euc, i.std.test[,-5])
i.pam7.euc.pred <- predict(i.pam7.euc, i.std.test[,-5])

i.pam2.man.pred <- predict(i.pam2.man, i.std.test[,-5])
i.pam3.man.pred <- predict(i.pam3.man, i.std.test[,-5])
i.pam5.man.pred <- predict(i.pam5.man, i.std.test[,-5])
i.pam7.man.pred <- predict(i.pam7.man, i.std.test[,-5])
##############
#14-2-1.R
##############
diameter <- function(clustering, data, metric="euclidean", stand=FALSE)
{
  clusters <- sort(unique(clustering))
  dm <- as.matrix(daisy(data, metric, stand))
  `names<-`(sapply(clusters, function(d) max(dm[clustering==d,clustering==d])),
            clusters)
}

  # training set diameter
diameter(i.pam2.euc$clustering, i.std.train[,-5])
diameter(i.pam3.euc$clustering, i.std.train[,-5])
diameter(i.pam5.euc$clustering, i.std.train[,-5])
diameter(i.pam7.euc$clustering, i.std.train[,-5])

diameter(i.pam2.man$clustering, i.std.train[,-5], metric="manhattan")
diameter(i.pam3.man$clustering, i.std.train[,-5], metric="manhattan")
diameter(i.pam5.man$clustering, i.std.train[,-5], metric="manhattan")
diameter(i.pam7.man$clustering, i.std.train[,-5], metric="manhattan")

  # test set diameter
diameter(i.pam2.euc.pred, i.std.test[,-5])
diameter(i.pam3.euc.pred, i.std.test[,-5])
diameter(i.pam5.euc.pred, i.std.test[,-5])
diameter(i.pam7.euc.pred, i.std.test[,-5])

diameter(i.pam2.man.pred, i.std.test[,-5], metric="manhattan")
diameter(i.pam3.man.pred, i.std.test[,-5], metric="manhattan")
diameter(i.pam5.man.pred, i.std.test[,-5], metric="manhattan")
diameter(i.pam7.man.pred, i.std.test[,-5], metric="manhattan")
##############
#14-2-2.R
##############
separation <- function(clustering, data, metric="euclidean", stand=FALSE)
{
  clusters <- sort(unique(clustering))
  dm <- as.matrix(daisy(data, metric, stand))
  `names<-`(sapply(clusters, function(d) min(dm[clustering==d,clustering!=d])),
            clusters)
}

  # training set separation
separation(i.pam2.euc$clustering, i.std.train[,-5])
separation(i.pam3.euc$clustering, i.std.train[,-5])
separation(i.pam5.euc$clustering, i.std.train[,-5])
separation(i.pam7.euc$clustering, i.std.train[,-5])

separation(i.pam2.man$clustering, i.std.train[,-5], metric="manhattan")
separation(i.pam3.man$clustering, i.std.train[,-5], metric="manhattan")
separation(i.pam5.man$clustering, i.std.train[,-5], metric="manhattan")
separation(i.pam7.man$clustering, i.std.train[,-5], metric="manhattan")

  # test set separation
separation(i.pam2.euc.pred, i.std.test[,-5])
separation(i.pam3.euc.pred, i.std.test[,-5])
separation(i.pam5.euc.pred, i.std.test[,-5])
separation(i.pam7.euc.pred, i.std.test[,-5])

separation(i.pam2.man.pred, i.std.test[,-5], metric="manhattan")
separation(i.pam3.man.pred, i.std.test[,-5], metric="manhattan")
separation(i.pam5.man.pred, i.std.test[,-5], metric="manhattan")
separation(i.pam7.man.pred, i.std.test[,-5], metric="manhattan")
##############
#14-2-3.R
##############
isolation <- function(clustering, data, metric="euclidean", stand=FALSE)
{
  clusters <- sort(unique(clustering))
  dm <- as.matrix(daisy(data, metric, stand))
  diam <- diameter(clustering, data, metric, stand)
  sep <- separation(clustering, data, metric, stand)

  is <- sapply(clusters,
               function(d)
               if (all(apply(dm[clustering==d,clustering==d,drop=FALSE], 1, max)<
                         apply(dm[clustering==d,clustering!=d,drop=FALSE], 1, min)))
                 "L*"
               else if (diam[d]<sep[d])
                 "L"
               else
                 "no")
  `names<-`(factor(is, levels=c("no", "L", "L*")), clusters)
}

  # training set isolation
isolation(i.pam2.euc$clustering, i.std.train[,-5])
isolation(i.pam3.euc$clustering, i.std.train[,-5])
isolation(i.pam5.euc$clustering, i.std.train[,-5])
isolation(i.pam7.euc$clustering, i.std.train[,-5])

isolation(i.pam2.man$clustering, i.std.train[,-5], metric="manhattan")
isolation(i.pam3.man$clustering, i.std.train[,-5], metric="manhattan")
isolation(i.pam5.man$clustering, i.std.train[,-5], metric="manhattan")
isolation(i.pam7.man$clustering, i.std.train[,-5], metric="manhattan")

  # test set isolation
isolation(i.pam2.euc.pred, i.std.test[,-5])
isolation(i.pam3.euc.pred, i.std.test[,-5])
isolation(i.pam5.euc.pred, i.std.test[,-5])
isolation(i.pam7.euc.pred, i.std.test[,-5])

isolation(i.pam2.man.pred, i.std.test[,-5], metric="manhattan")
isolation(i.pam3.man.pred, i.std.test[,-5], metric="manhattan")
isolation(i.pam5.man.pred, i.std.test[,-5], metric="manhattan")
isolation(i.pam7.man.pred, i.std.test[,-5], metric="manhattan")
##############
#14-2-4.R
##############
silwidth <- function(clustering, d, data, metric="euclidean", stand=FALSE)
{
  if (sum(clustering==d)==1)
    1  # singleton cluster
  else
  {
    clusters <- unique(clustering)
    other <- clusters[! clusters %in% d]
    dm <- as.matrix(daisy(data, metric, stand))
    avg.intra <- apply(dm[clustering==d,clustering==d,drop=FALSE], 1, sum)/
                   (sum(clustering==d)-1)
    avg.inter <- apply(sapply(other,
                              function(d1)
                              apply(dm[clustering==d,clustering==d1,drop=FALSE],
                                    1, mean)),
                       1, min)
    (avg.inter-avg.intra)/pmax(avg.inter, avg.intra)
  }
}

silwidth.cluster <- function(clustering, data, metric="euclidean", stand=FALSE)
{
  clusters <- sort(unique(clustering))
  `names<-`(sapply(clusters, function(d)
                             mean(silwidth(clustering, d, data, metric, stand))),
            clusters)
}

  # training set per-cluster silhouette width
silwidth.cluster(i.pam2.euc$clustering, i.std.train[,-5])
silwidth.cluster(i.pam3.euc$clustering, i.std.train[,-5])
silwidth.cluster(i.pam5.euc$clustering, i.std.train[,-5])
silwidth.cluster(i.pam7.euc$clustering, i.std.train[,-5])

silwidth.cluster(i.pam2.man$clustering, i.std.train[,-5], metric="manhattan")
silwidth.cluster(i.pam3.man$clustering, i.std.train[,-5], metric="manhattan")
silwidth.cluster(i.pam5.man$clustering, i.std.train[,-5], metric="manhattan")
silwidth.cluster(i.pam7.man$clustering, i.std.train[,-5], metric="manhattan")

  # test set per-cluster silhouette width
silwidth.cluster(i.pam2.euc.pred, i.std.test[,-5])
silwidth.cluster(i.pam3.euc.pred, i.std.test[,-5])
silwidth.cluster(i.pam5.euc.pred, i.std.test[,-5])
silwidth.cluster(i.pam7.euc.pred, i.std.test[,-5])

silwidth.cluster(i.pam2.man.pred, i.std.test[,-5], metric="manhattan")
silwidth.cluster(i.pam3.man.pred, i.std.test[,-5], metric="manhattan")
silwidth.cluster(i.pam5.man.pred, i.std.test[,-5], metric="manhattan")
silwidth.cluster(i.pam7.man.pred, i.std.test[,-5], metric="manhattan")

  # training set silhouette plots
par(mfrow=c(2, 1), mar=c(2, 6, 0, 1), oma=c(0, 0, 2, 0))
barplot(sort(silwidth(i.pam2.euc$clustering, 1, i.std.train[,-5])),
        xlim=c(-0.2, 1), yaxt="n", horiz=TRUE)
barplot(sort(silwidth(i.pam2.euc$clustering, 2, i.std.train[,-5])),
        xlim=c(-0.2, 1), yaxt="n", horiz=TRUE)
title("Training set, k=2", outer=TRUE)

par(mfrow=c(3, 1), mar=c(2, 6, 0, 1), oma=c(0, 0, 2, 0))
barplot(sort(silwidth(i.pam3.euc$clustering, 1, i.std.train[,-5])),
        xlim=c(-0.2, 1), yaxt="n", horiz=TRUE)
barplot(sort(silwidth(i.pam3.euc$clustering, 2, i.std.train[,-5])),
        xlim=c(-0.2, 1), yaxt="n", horiz=TRUE)
barplot(sort(silwidth(i.pam3.euc$clustering, 3, i.std.train[,-5])),
        xlim=c(-0.2, 1), yaxt="n", horiz=TRUE)
title("Training set, k=3", outer=TRUE)

  # test set silhouette plot
par(mfrow=c(2, 1), mar=c(2, 6, 0, 1), oma=c(0, 0, 2, 0))
barplot(sort(silwidth(i.pam2.euc.pred, 1, i.std.test[,-5])),
        xlim=c(-0.2, 1), yaxt="n", horiz=TRUE)
barplot(sort(silwidth(i.pam2.euc.pred, 2, i.std.test[,-5])),
        xlim=c(-0.2, 1), yaxt="n", horiz=TRUE)
title("Test set, k=2", outer=TRUE)

par(mfrow=c(3, 1), mar=c(2, 6, 0, 1), oma=c(0, 0, 2, 0))
barplot(sort(silwidth(i.pam3.euc.pred, 1, i.std.test[,-5])),
        xlim=c(-0.2, 1), yaxt="n", horiz=TRUE)
barplot(sort(silwidth(i.pam3.euc.pred, 2, i.std.test[,-5])),
        xlim=c(-0.2, 1), yaxt="n", horiz=TRUE)
barplot(sort(silwidth(i.pam3.euc.pred, 3, i.std.test[,-5])),
        xlim=c(-0.2, 1), yaxt="n", horiz=TRUE)
title("Test set, k=3", outer=TRUE)
##############
#14-2-5.R
##############
dbindex <- function(clustering, centers, data, metric="euclidean", stand=FALSE)
{
  clusters <- sort(unique(clustering))
  ds <- as.matrix(daisy(rbind(data, centers), metric, stand))

  to.center <- sapply(clusters, function(d) mean(ds[clustering==d,nrow(data)+d]))
  between.centers <- ds[(nrow(data)+1):(nrow(data)+length(clusters)),
                        (nrow(data)+1):(nrow(data)+length(clusters))]
  diag(between.centers) <- NA

  `dimnames<-`(outer(clusters, clusters, Vectorize(function(d1, d2)
                                                   (to.center[d1]+to.center[d2])/
                                                     between.centers[d1,d2])),
               list(clusters, clusters))
}

dbindex.cluster <- function(clustering, centers, data,
                            metric="euclidean", stand=FALSE)
{ apply(dbindex(clustering, centers, data, metric, stand), 1, max, na.rm=TRUE) }

  # training set Davies-Bouldin index for cluster pairs
dbindex(i.pam3.euc$clustering, i.pam3.euc$medoids, i.std.train[,-5])

  # training set Davies-Bouldin index
dbindex.cluster(i.pam2.euc$clustering, i.pam2.euc$medoids, i.std.train[,-5])
dbindex.cluster(i.pam3.euc$clustering, i.pam3.euc$medoids, i.std.train[,-5])
dbindex.cluster(i.pam5.euc$clustering, i.pam5.euc$medoids, i.std.train[,-5])
dbindex.cluster(i.pam7.euc$clustering, i.pam7.euc$medoids, i.std.train[,-5])

dbindex.cluster(i.pam2.man$clustering, i.pam2.euc$medoids, i.std.train[,-5],
                metric="manhattan")
dbindex.cluster(i.pam3.man$clustering, i.pam3.euc$medoids, i.std.train[,-5],
                metric="manhattan")
dbindex.cluster(i.pam5.man$clustering, i.pam5.euc$medoids, i.std.train[,-5],
                metric="manhattan")
dbindex.cluster(i.pam7.man$clustering, i.pam7.euc$medoids, i.std.train[,-5],
                metric="manhattan")

  # test set Davies-Bouldin index
dbindex.cluster(i.pam2.euc.pred, i.pam2.euc$medoids, i.std.test[,-5])
dbindex.cluster(i.pam3.euc.pred, i.pam3.euc$medoids, i.std.test[,-5])
dbindex.cluster(i.pam5.euc.pred, i.pam5.euc$medoids, i.std.test[,-5])
dbindex.cluster(i.pam7.euc.pred, i.pam7.euc$medoids, i.std.test[,-5])

dbindex.cluster(i.pam2.man.pred, i.pam2.euc$medoids, i.std.test[,-5],
                metric="manhattan")
dbindex.cluster(i.pam3.man.pred, i.pam3.euc$medoids, i.std.test[,-5],
                metric="manhattan")
dbindex.cluster(i.pam5.man.pred, i.pam5.euc$medoids, i.std.test[,-5],
                metric="manhattan")
dbindex.cluster(i.pam7.man.pred, i.pam7.euc$medoids, i.std.test[,-5],
                metric="manhattan")
##############
#14-3-1.R
##############
dunn <- function(clustering, data, metric="euclidean", stand=FALSE)
{
  min(separation(clustering, data, metric, stand))/
    max(diameter(clustering, data, metric, stand))
}

  # training set Dunn index
dunn(i.pam2.euc$clustering, i.std.train[,-5])
dunn(i.pam3.euc$clustering, i.std.train[,-5])
dunn(i.pam5.euc$clustering, i.std.train[,-5])
dunn(i.pam7.euc$clustering, i.std.train[,-5])

dunn(i.pam2.man$clustering, i.std.train[,-5], metric="manhattan")
dunn(i.pam3.man$clustering, i.std.train[,-5], metric="manhattan")
dunn(i.pam5.man$clustering, i.std.train[,-5], metric="manhattan")
dunn(i.pam7.man$clustering, i.std.train[,-5], metric="manhattan")

  # test set Dunn index
dunn(i.pam2.euc.pred, i.std.test[,-5])
dunn(i.pam3.euc.pred, i.std.test[,-5])
dunn(i.pam5.euc.pred, i.std.test[,-5])
dunn(i.pam7.euc.pred, i.std.test[,-5])

dunn(i.pam2.man.pred, i.std.test[,-5], metric="manhattan")
dunn(i.pam3.man.pred, i.std.test[,-5], metric="manhattan")
dunn(i.pam5.man.pred, i.std.test[,-5], metric="manhattan")
dunn(i.pam7.man.pred, i.std.test[,-5], metric="manhattan")
##############
#14-3-2.R
##############
dbindex.avg <- function(clustering, centers, data, metric="euclidean", stand=FALSE)
{ mean(dbindex.cluster(clustering, centers, data, metric, stand)) }

  # training set average Davies-Bouldin index
dbindex.avg(i.pam2.euc$clustering, i.pam2.euc$medoids, i.std.train[,-5])
dbindex.avg(i.pam3.euc$clustering, i.pam3.euc$medoids, i.std.train[,-5])
dbindex.avg(i.pam5.euc$clustering, i.pam5.euc$medoids, i.std.train[,-5])
dbindex.avg(i.pam7.euc$clustering, i.pam7.euc$medoids, i.std.train[,-5])

dbindex.avg(i.pam2.man$clustering, i.pam2.euc$medoids, i.std.train[,-5],
            metric="manhattan")
dbindex.avg(i.pam3.man$clustering, i.pam3.euc$medoids, i.std.train[,-5],
            metric="manhattan")
dbindex.avg(i.pam5.man$clustering, i.pam5.euc$medoids, i.std.train[,-5],
            metric="manhattan")
dbindex.avg(i.pam7.man$clustering, i.pam7.euc$medoids, i.std.train[,-5],
            metric="manhattan")

  # test set average Davies-Bouldin index
dbindex.avg(i.pam2.euc.pred, i.pam2.euc$medoids, i.std.test[,-5])
dbindex.avg(i.pam3.euc.pred, i.pam3.euc$medoids, i.std.test[,-5])
dbindex.avg(i.pam5.euc.pred, i.pam5.euc$medoids, i.std.test[,-5])
dbindex.avg(i.pam7.euc.pred, i.pam7.euc$medoids, i.std.test[,-5])

dbindex.avg(i.pam2.man.pred, i.pam2.euc$medoids, i.std.test[,-5], metric="manhattan")
dbindex.avg(i.pam3.man.pred, i.pam3.euc$medoids, i.std.test[,-5], metric="manhattan")
dbindex.avg(i.pam5.man.pred, i.pam5.euc$medoids, i.std.test[,-5], metric="manhattan")
dbindex.avg(i.pam7.man.pred, i.pam7.euc$medoids, i.std.test[,-5], metric="manhattan")
##############
#14-3-3.R
##############
cindex <- function(clustering, data, metric="euclidean", stand=FALSE)
{
  clusters <- unique(clustering)
  dm <- as.matrix(daisy(data, metric, stand))
  dm[lower.tri(dm)] <- diag(dm) <- NA
  sdm <- sort(dm)
  cc <- table(clustering)
  m <- sum(cc*(cc-1)/2)

  s <- sum(sapply(clusters,
                  function(d) sum(dm[clustering==d,clustering==d], na.rm=TRUE)))
  smin <- sum(sdm[1:m])
  smax <- sum(sdm[(length(sdm)-m+1):length(sdm)])
  (s-smin)/(smax-smin)
}

  # training set C index
cindex(i.pam2.euc$clustering, i.std.train[,-5])
cindex(i.pam3.euc$clustering, i.std.train[,-5])
cindex(i.pam5.euc$clustering, i.std.train[,-5])
cindex(i.pam7.euc$clustering, i.std.train[,-5])

cindex(i.pam2.man$clustering, i.std.train[,-5], metric="manhattan")
cindex(i.pam3.man$clustering, i.std.train[,-5], metric="manhattan")
cindex(i.pam5.man$clustering, i.std.train[,-5], metric="manhattan")
cindex(i.pam7.man$clustering, i.std.train[,-5], metric="manhattan")

  # test set C index
cindex(i.pam2.euc.pred, i.std.test[,-5])
cindex(i.pam3.euc.pred, i.std.test[,-5])
cindex(i.pam5.euc.pred, i.std.test[,-5])
cindex(i.pam7.euc.pred, i.std.test[,-5])

cindex(i.pam2.man.pred, i.std.test[,-5], metric="manhattan")
cindex(i.pam3.man.pred, i.std.test[,-5], metric="manhattan")
cindex(i.pam5.man.pred, i.std.test[,-5], metric="manhattan")
cindex(i.pam7.man.pred, i.std.test[,-5], metric="manhattan")
##############
#14-3-4.R
##############
silwidth.avg <- function(clustering, data, metric="euclidean", stand=FALSE)
{
  clusters <- unique(clustering)
  mean(unlist(sapply(clusters,
                     function(d) silwidth(clustering, d, data, metric, stand))))
}

  # training set average silhouette width
silwidth.avg(i.pam2.euc$clustering, i.std.train[,-5])
silwidth.avg(i.pam3.euc$clustering, i.std.train[,-5])
silwidth.avg(i.pam5.euc$clustering, i.std.train[,-5])
silwidth.avg(i.pam7.euc$clustering, i.std.train[,-5])

silwidth.avg(i.pam2.man$clustering, i.std.train[,-5], metric="manhattan")
silwidth.avg(i.pam3.man$clustering, i.std.train[,-5], metric="manhattan")
silwidth.avg(i.pam5.man$clustering, i.std.train[,-5], metric="manhattan")
silwidth.avg(i.pam7.man$clustering, i.std.train[,-5], metric="manhattan")

  # test set average silhouette width
silwidth.avg(i.pam2.euc.pred, i.std.test[,-5])
silwidth.avg(i.pam3.euc.pred, i.std.test[,-5])
silwidth.avg(i.pam5.euc.pred, i.std.test[,-5])
silwidth.avg(i.pam7.euc.pred, i.std.test[,-5])

silwidth.avg(i.pam2.man.pred, i.std.test[,-5], metric="manhattan")
silwidth.avg(i.pam3.man.pred, i.std.test[,-5], metric="manhattan")
silwidth.avg(i.pam5.man.pred, i.std.test[,-5], metric="manhattan")
silwidth.avg(i.pam7.man.pred, i.std.test[,-5], metric="manhattan")
##############
#14-3-5.R
##############
clusloglik <- function(train.clustering, train.data,
                       eval.clustering=train.clustering, eval.data=train.data)
{
  clusters <- unique(train.clustering)
  prob.d <- function(d) { sum(train.clustering==d)/nrow(train.data) }
  prob.avd <- function(a, v, d)
  {
    ifelse(is.numeric(v),
           dnorm(v, mmean(train.data[train.clustering==d,a],
                          m0=mean(train.data[,a])),
                 sqrt(mvar(train.data[train.clustering==d,a],
                           m0=mean(train.data[,a]), s02=var(train.data[,a])))),
           mest(sum(train.data[,a]==v & train.clustering==d),
                sum(train.clustering==d),
                nlevels(train.data[,a]), 1/nlevels(train.data[,a])))
  }

  sum(sapply(1:nrow(eval.data),
             function(i)
             log(sum(sapply(clusters,
                            function(d)
                            prob.d(d)*prod(mapply(function(a, v) prob.avd(a, v, d),
                                                  1:ncol(eval.data),
                                                  eval.data[i,])))))))
}

  # training set loglikelihood
clusloglik(i.pam2.euc$clustering, i.std.train[,-5])
clusloglik(i.pam3.euc$clustering, i.std.train[,-5])
clusloglik(i.pam5.euc$clustering, i.std.train[,-5])
clusloglik(i.pam7.euc$clustering, i.std.train[,-5])

clusloglik(i.pam2.man$clustering, i.std.train[,-5])
clusloglik(i.pam3.man$clustering, i.std.train[,-5])
clusloglik(i.pam5.man$clustering, i.std.train[,-5])
clusloglik(i.pam7.man$clustering, i.std.train[,-5])

  # test set loglikelihood
clusloglik(i.pam2.euc$clustering, i.std.train[,-5], i.pam2.euc.pred,
           i.std.test[,-5])
clusloglik(i.pam3.euc$clustering, i.std.train[,-5], i.pam3.euc.pred,
           i.std.test[,-5])
clusloglik(i.pam5.euc$clustering, i.std.train[,-5], i.pam5.euc.pred,
           i.std.test[,-5])
clusloglik(i.pam7.euc$clustering, i.std.train[,-5], i.pam7.euc.pred,
           i.std.test[,-5])

clusloglik(i.pam2.man$clustering, i.std.train[,-5], i.pam2.man.pred,
           i.std.test[,-5])
clusloglik(i.pam3.man$clustering, i.std.train[,-5], i.pam3.man.pred,
           i.std.test[,-5])
clusloglik(i.pam5.man$clustering, i.std.train[,-5], i.pam5.man.pred,
           i.std.test[,-5])
clusloglik(i.pam7.man$clustering, i.std.train[,-5], i.pam7.man.pred,
           i.std.test[,-5])
##############
#14-4-1.R
##############
clustclas <- function(train.clustering, train.classes,
                      eval.clustering=train.clustering)
{
  clusters <- unique(train.clustering)
  labels <-
    sapply(clusters,
         function(d)
         levels(train.classes)[which.max(pdisc(train.classes[train.clustering==d]))])
  factor(labels[eval.clustering], levels=levels(train.classes))
}

  # training set error
err(clustclas(i.pam2.euc$clustering, i.std.train[,5]), i.std.train[,5])
err(clustclas(i.pam3.euc$clustering, i.std.train[,5]), i.std.train[,5])
err(clustclas(i.pam5.euc$clustering, i.std.train[,5]), i.std.train[,5])
err(clustclas(i.pam7.euc$clustering, i.std.train[,5]), i.std.train[,5])

err(clustclas(i.pam2.man$clustering, i.std.train[,5]), i.std.train[,5])
err(clustclas(i.pam3.man$clustering, i.std.train[,5]), i.std.train[,5])
err(clustclas(i.pam5.man$clustering, i.std.train[,5]), i.std.train[,5])
err(clustclas(i.pam7.man$clustering, i.std.train[,5]), i.std.train[,5])

  # test set error
err(clustclas(i.pam2.euc$clustering, i.std.train[,5], i.pam2.euc.pred),
    i.std.test[,5])
err(clustclas(i.pam3.euc$clustering, i.std.train[,5], i.pam3.euc.pred),
    i.std.test[,5])
err(clustclas(i.pam5.euc$clustering, i.std.train[,5], i.pam5.euc.pred),
    i.std.test[,5])
err(clustclas(i.pam7.euc$clustering, i.std.train[,5], i.pam7.euc.pred),
    i.std.test[,5])

err(clustclas(i.pam2.man$clustering, i.std.train[,5], i.pam2.man.pred),
    i.std.test[,5])
err(clustclas(i.pam3.man$clustering, i.std.train[,5], i.pam3.man.pred),
    i.std.test[,5])
err(clustclas(i.pam5.man$clustering, i.std.train[,5], i.pam5.man.pred),
    i.std.test[,5])
err(clustclas(i.pam7.man$clustering, i.std.train[,5], i.pam7.man.pred),
    i.std.test[,5])
##############
#14-4-2.R
##############
randindex <- function(clustering, classes)
{
  mean(outer(1:length(clustering), 1:length(classes),
             function(i, j)
             ifelse(i!=j,
                    clustering[i]==clustering[j] & classes[i]==classes[j] |
                    clustering[i]!=clustering[j] & classes[i]!=classes[j], NA)),
       na.rm=TRUE)
}

  # training set Rand index
randindex(i.pam2.euc$clustering, i.std.train[,5])
randindex(i.pam3.euc$clustering, i.std.train[,5])
randindex(i.pam5.euc$clustering, i.std.train[,5])
randindex(i.pam7.euc$clustering, i.std.train[,5])

randindex(i.pam2.man$clustering, i.std.train[,5])
randindex(i.pam3.man$clustering, i.std.train[,5])
randindex(i.pam5.man$clustering, i.std.train[,5])
randindex(i.pam7.man$clustering, i.std.train[,5])

  # test set Rand index
randindex(i.pam2.man$clustering, i.std.test[,5])
randindex(i.pam3.man$clustering, i.std.test[,5])
randindex(i.pam5.man$clustering, i.std.test[,5])
randindex(i.pam7.man$clustering, i.std.test[,5])

randindex(i.pam2.man.pred, i.std.test[,5])
randindex(i.pam3.man.pred, i.std.test[,5])
randindex(i.pam5.man.pred, i.std.test[,5])
randindex(i.pam7.man.pred, i.std.test[,5])
##############
#14-4-3.R
##############
  # training set chi-square
chisq.test(i.pam2.euc$clustering, i.std.train[,5])
chisq.test(i.pam3.euc$clustering, i.std.train[,5])
chisq.test(i.pam5.euc$clustering, i.std.train[,5])
chisq.test(i.pam7.euc$clustering, i.std.train[,5])

chisq.test(i.pam2.man$clustering, i.std.train[,5])
chisq.test(i.pam3.man$clustering, i.std.train[,5])
chisq.test(i.pam5.man$clustering, i.std.train[,5])
chisq.test(i.pam7.man$clustering, i.std.train[,5])

  # test set chi-square
chisq.test(i.pam2.man.pred, i.std.test[,5])
chisq.test(i.pam3.man.pred, i.std.test[,5])
chisq.test(i.pam5.man.pred, i.std.test[,5])
chisq.test(i.pam7.man.pred, i.std.test[,5])

chisq.test(i.pam2.man.pred, i.std.test[,5])
chisq.test(i.pam3.man.pred, i.std.test[,5])
chisq.test(i.pam5.man.pred, i.std.test[,5])
chisq.test(i.pam7.man.pred, i.std.test[,5])

  # training set mutual information
mutinfo(i.pam2.euc$clustering, i.std.train[,5])
mutinfo(i.pam3.euc$clustering, i.std.train[,5])
mutinfo(i.pam5.euc$clustering, i.std.train[,5])
mutinfo(i.pam7.euc$clustering, i.std.train[,5])

mutinfo(i.pam2.man$clustering, i.std.train[,5])
mutinfo(i.pam3.man$clustering, i.std.train[,5])
mutinfo(i.pam5.man$clustering, i.std.train[,5])
mutinfo(i.pam7.man$clustering, i.std.train[,5])

  # test set mutual information
mutinfo(i.pam2.man.pred, i.std.test[,5])
mutinfo(i.pam3.man.pred, i.std.test[,5])
mutinfo(i.pam5.man.pred, i.std.test[,5])
mutinfo(i.pam7.man.pred, i.std.test[,5])

mutinfo(i.pam2.man.pred, i.std.test[,5])
mutinfo(i.pam3.man.pred, i.std.test[,5])
mutinfo(i.pam5.man.pred, i.std.test[,5])
mutinfo(i.pam7.man.pred, i.std.test[,5])
##############
#15-1-1.R
##############
library(dmr.claseval)
library(dmr.dectree)
library(dmr.regeval)
library(dmr.regtree)
library(dmr.stats)
library(dmr.util)

library(rpart)
library(e1071)

data(HouseVotes84, package="mlbench")
data(BostonHousing, package="mlbench")

set.seed(12)

rhv <- runif(nrow(HouseVotes84))
hv.train <- HouseVotes84[rhv>=0.33,]
hv.test <- HouseVotes84[rhv<0.33,]

rbh <- runif(nrow(BostonHousing))
bh.train <- BostonHousing[rbh>=0.33,]
bh.test <- BostonHousing[rbh<0.33,]

hv.tree <- rpart(Class~., hv.train)
hv.nb <- naiveBayes(Class~., hv.train)

hv.err.tree <- err(predict(hv.tree, hv.test, type="c"), hv.test$Class)
hv.err.nb <- err(predict(hv.nb, hv.test), hv.test$Class)

bh.tree <- rpart(medv~., bh.train)
bh.lm <- lm(medv~., bh.train)

bh.mse.tree <- mse(predict(bh.tree, bh.test), bh.test$medv)
bh.mse.lm <- mse(predict(bh.lm, bh.test), bh.test$medv)
##############
#15-3-1.R
##############
## generate base models by instance sampling
base.ensemble.sample.x <- function(formula, data, m, alg, args=NULL,
                                   size=nrow(data), replace=TRUE)
{
  lapply(1:m, function(i)
              {
                bag <- sample(nrow(data), size=nrow(data), replace=replace)
                do.call(alg, c(list(formula, data[bag,]), args))
              })
}

  # base models for the HouseVotes84 data
hv.bm.tree.sx <- base.ensemble.sample.x(Class~., hv.train, 50, rpart)
hv.bm.nb.sx <- base.ensemble.sample.x(Class~., hv.train, 50, naiveBayes)

  # base models for the BostonHousing data
bh.bm.tree.sx <- base.ensemble.sample.x(medv~., bh.train, 50, rpart)
bh.bm.lm.sx <- base.ensemble.sample.x(medv~., bh.train, 50, lm)

  # base model training set errors for the HouseVotes84 data
hv.train.err.tree.sx <- sapply(hv.bm.tree.sx,
                               function(h) err(predict(h, hv.train, type="c"),
                                               hv.train$Class))
hv.train.err.nb.sx <- sapply(hv.bm.nb.sx,
                             function(h) err(predict(h, hv.train), hv.train$Class))

  # base model training set MSE values for the BostonHousing data
bh.train.mse.tree.sx <- sapply(bh.bm.tree.sx,
                               function(h) mse(predict(h, bh.train), bh.train$medv))
bh.train.mse.lm.sx <- sapply(bh.bm.lm.sx,
                             function(h) mse(predict(h, bh.train), bh.train$medv))

  # base model test set errors for the HouseVotes84 data
hv.test.err.tree.sx <- sapply(hv.bm.tree.sx,
                              function(h) err(predict(h, hv.test, type="c"),
                                              hv.test$Class))
hv.test.err.nb.sx <- sapply(hv.bm.nb.sx,
                            function(h) err(predict(h, hv.test), hv.test$Class))

  # base model test set MSE values for the BostonHousing data
bh.test.mse.tree.sx <- sapply(bh.bm.tree.sx,
                              function(h) mse(predict(h, bh.test), bh.test$medv))
bh.test.mse.lm.sx <- sapply(bh.bm.lm.sx,
                            function(h) mse(predict(h, bh.test), bh.test$medv))
##############
#15-3-2.R
##############
## generate base models by instance weighting
base.ensemble.weight.x <- function(formula, data, m, alg, args=NULL,
                                   weights=runif(nrow(data), min=0.3, max=3),
                                   reweight=function(w, p=NULL)
                                            runif(nrow(data), min=0.3, max=3),
                                   predf=predict)
{
  skip.cond(lapply(1:m,
                   function(i)
                   {
                     if (!is.null(weights))
                     {
                       h <- do.call(alg, c(list(formula, data, weights=weights),
                                           args))
                       pred <- predf(h, data)
                       if (!is.null(weights <<- reweight(weights, pred)))
                         h
                     }
                   }),
            is.null)
}

  # base models for the HouseVotes84 data
hv.bm.tree.wx <- base.ensemble.weight.x(Class~., hv.train, 50, rpart)

  # base models for the BostonHousing data
bh.bm.tree.wx <- base.ensemble.weight.x(medv~., bh.train, 50, rpart)
bh.bm.lm.wx <- base.ensemble.weight.x(medv~., bh.train, 50, lm)

  # base model training set errors for the HouseVotes84 data
hv.train.err.tree.wx <- sapply(hv.bm.tree.wx,
                               function(h) err(predict(h, hv.train, type="c"),
                                               hv.train$Class))

  # base model training set MSE values for the BostonHousing data
bh.train.mse.tree.wx <- sapply(bh.bm.tree.wx,
                               function(h) mse(predict(h, bh.train), bh.train$medv))
bh.train.mse.lm.wx <- sapply(bh.bm.lm.wx,
                             function(h) mse(predict(h, bh.train), bh.train$medv))

  # base model test set errors for the HouseVotes84 data
hv.test.err.tree.wx <- sapply(hv.bm.tree.wx,
                              function(h) err(predict(h, hv.test, type="c"),
                                              hv.test$Class))

  # base model test set MSE values for the BostonHousing data
bh.test.mse.tree.wx <- sapply(bh.bm.tree.wx,
                              function(h) mse(predict(h, bh.test), bh.test$medv))
bh.test.mse.lm.wx <- sapply(bh.bm.lm.wx,
                            function(h) mse(predict(h, bh.test), bh.test$medv))
##############
#15-3-3.R
##############
## generate base models by attribute sampling
base.ensemble.sample.a <- function(formula, data, m, alg, args=NULL,
                                   frac=0, replace=TRUE)
{
  attributes <- x.vars(formula, data)
  target <- y.var(formula)
  ns <- ifelse(clip.val(frac, 0, 1)>0, ceiling(frac*length(attributes)),
                                       ceiling(sqrt(length(attributes))))
  lapply(1:m, function(i)
              {
                sa <- sample(length(attributes), ns)
                do.call(alg, c(list(make.formula(target, attributes[sa]), data),
                               args))
              })
}

  # base models for the HouseVotes84 data
hv.bm.tree.sa <- base.ensemble.sample.a(Class~., hv.train, 50, rpart,
                                        args=list(minsplit=2, cp=0))
hv.bm.nb.sa <- base.ensemble.sample.a(Class~., hv.train, 50, naiveBayes)

  # base models for the BostonHousing data
bh.bm.tree.sa <- base.ensemble.sample.a(medv~., bh.train, 50, rpart,
                                        args=list(minsplit=2, cp=0))
bh.bm.lm.sa <- base.ensemble.sample.a(medv~., bh.train, 50, lm)

  # base model training set errors for the HouseVotes84 data
hv.train.err.tree.sa <- sapply(hv.bm.tree.sa,
                               function(h) err(predict(h, hv.train, type="c"),
                                               hv.train$Class))
hv.train.err.nb.sa <- sapply(hv.bm.nb.sa,
                             function(h) err(predict(h, hv.train), hv.train$Class))

  # base model training set MSE values for the BostonHousing data
bh.train.mse.tree.sa <- sapply(bh.bm.tree.sa,
                               function(h) mse(predict(h, bh.train), bh.train$medv))
bh.train.mse.lm.sa <- sapply(bh.bm.lm.sa,
                             function(h) mse(predict(h, bh.train), bh.train$medv))

  # base model test set errors for the HouseVotes84 data
hv.test.err.tree.sa <- sapply(hv.bm.tree.sa,
                              function(h) err(predict(h, hv.test, type="c"),
                                              hv.test$Class))
hv.test.err.nb.sa <- sapply(hv.bm.nb.sa,
                            function(h) err(predict(h, hv.test), hv.test$Class))

  # base model test set MSE values for the BostonHousing data
bh.test.mse.tree.sa <- sapply(bh.bm.tree.sa,
                              function(h) mse(predict(h, bh.test), bh.test$medv))
bh.test.mse.lm.sa <- sapply(bh.bm.lm.sa,
                            function(h) mse(predict(h, bh.test), bh.test$medv))
##############
#15-3-4.R
##############
## randomized decision tree growing
## with split selection using ns randomly chosen attributes at each node
## (if unspecified or 0, it defaults to the square root of the number of attributes)
grow.randdectree <- function(formula, data, ns=0,
                             imp=entropy.p, maxprob=0.999, minsplit=2, maxdepth=8)
{
  init <- function()
  {
    clabs <<- factor(levels(data[[class]]),
                     levels=levels(data[[class]]))  # class labels
    tree <<- data.frame(node=1, attribute=NA, value=NA, class=NA, count=NA,
                        `names<-`(rep(list(NA), length(clabs)),
                                  paste("p", clabs, sep=".")))
    cprobs <<- (ncol(tree)-length(clabs)+1):ncol(tree)  # class probability columns
    nodemap <<- rep(1, nrow(data))
    n <<- 1
  }

  next.node <- function(node)
  {
    if (any(opn <- tree$node>node))
      min(tree$node[opn])
    else Inf
  }

  class.distribution <- function(node)
  {
    tree[tree$node==node,"count"] <<- sum(nodemap==node)
    tree[tree$node==node,cprobs] <<- pdisc(data[nodemap==node,class])
  }

  class.label <- function(node)
  {
    tree$class[tree$node==node] <<- which.max(tree[tree$node==node,cprobs])
  }

  stop.criteria <- function(node)
  {
    node>=2^maxdepth || tree[tree$node==node,"count"]<minsplit ||
      max(tree[tree$node==node,cprobs])>maxprob
  }

  split.eval <- function(av, sv, cl)
  {
    cond <- !is.na(av) & (if (is.numeric(av)) av<=as.numeric(sv) else av==sv)

    pd1 <- pdisc(cl[cond])
    n1 <- sum(cond)
    pd0 <- pdisc(cl[!cond])
    n0 <- sum(!cond)

    if (n1>0 && n0>0)
      weighted.impurity(pd1, n1, pd0, n0, imp)
    else
      Inf
  }

  split.select <- function(node)
  {
    splits <- data.frame()
    for (attribute in sample(attributes, ns))
    {
      uav <- sort(unique(data[nodemap==node,attribute]))
      if (length(uav)>1)
        splits <- rbind(splits,
                        data.frame(attribute=attribute,
                                   value=if (is.numeric(uav))
                                           midbrk(uav)
                                         else as.character(uav),
                                   stringsAsFactors=F))
    }

    if (nrow(splits)>0)
      splits$eval <- sapply(1:nrow(splits),
                            function(s)
                            split.eval(data[nodemap==node,splits$attribute[s]],
                                       splits$value[s],
                                       data[nodemap==node,class]))
    if ((best.eval <- min(splits$eval))<Inf)
      tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
    return(best.eval)
  }

  split.apply <- function(node)
  {
    tree <<- rbind(tree,
                   data.frame(node=(2*node):(2*node+1),
                              attribute=NA, value=NA, class=NA, count=NA,
                              `names<-`(rep(list(NA), length(clabs)),
                                        paste("p", clabs, sep="."))))

    av <- data[[tree$attribute[tree$node==node]]]
    cond <- !is.na(av) & (if (is.numeric(av))
                            av<=as.numeric(tree$value[tree$node==node])
                          else av==tree$value[tree$node==node])
    nodemap[nodemap==node & cond] <<- 2*node
    nodemap[nodemap==node & !cond] <<- 2*node+1
  }

  tree <- nodemap <- node <- NULL
  clabs <- cprobs <- NULL
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  ns <- ifelse(ns==0, round(sqrt(length(attributes))),
                      clip.val(ns, 1, length(attributes)))

  init()
  while (is.finite(node))
  {
    class.distribution(node)
    class.label(node)
    if (!stop.criteria(node))
      if (split.select(node)<Inf)
        split.apply(node)
    node <- next.node(node)
  }
  tree$class <- clabs[tree$class]
  `class<-`(tree, "dectree")
}


## randomized regression tree growing
## with split selection using ns randomly chosen attributes at each node
## (if unspecified or 0, it defaults to the square root of the number of attributes)
grow.randregtree <- function(formula, data, ns=0,
                             minvar=0.005, minsplit=2, maxdepth=8)
{
  init <- function()
  {
    tree <<- data.frame(node=1, attribute=NA, value=NA, target=NA,
                        count=NA, mean=NA, variance=NA)
    nodemap <<- rep(1, nrow(data))
    node <<- 1
  }

  next.node <- function(node)
  {
    if (any(opn <- tree$node>node))
      min(tree$node[opn])
    else Inf
  }

  target.summary <- function(node)
  {
    tree$count[tree$node==node] <<- sum(nodemap==node)
    tree$mean[tree$node==node] <<- mean(data[nodemap==node,target])
    tree$variance[tree$node==node] <<- var1(data[nodemap==node,target])
  }

  target.value <- function(node)
  {
    tree$target[tree$node==node] <<- tree$mean[tree$node==node]
  }

  stop.criteria <- function(node)
  {
    node>=2^maxdepth || tree$count[tree$node==node]<minsplit ||
      tree$variance[tree$node==node]<minvar
  }

  split.eval <- function(av, sv, tv)
  {
    cond <- !is.na(av) & (if (is.numeric(av)) av<=as.numeric(sv) else av==sv)
    v1 <- tv[cond]
    n1 <- sum(cond)
    v0 <- tv[!cond]
    n0 <- sum(!cond)
    if (n1>0 && n0>0)
      weighted.dispersion(v1, v0)
    else
      Inf
  }

  split.select <- function(node)
  {
    splits <- data.frame()
    for (attribute in sample(attributes, ns))
    {
      uav <- sort(unique(data[nodemap==node,attribute]))
      if (length(uav)>1)
        splits <- rbind(splits,
                        data.frame(attribute=attribute,
                                   value=if (is.numeric(uav))
                                           midbrk(uav)
                                         else as.character(uav),
                                   stringsAsFactors=F))
    }

    if (nrow(splits)>0)
      splits$eval <- sapply(1:nrow(splits),
                            function(s)
                            split.eval(data[nodemap==node,splits$attribute[s]],
                                       splits$value[s],
                                       data[nodemap==node,target]))
    if ((best.eval <- min(splits$eval))<Inf)
      tree[tree$node==node,2:3] <<- splits[which.min(splits$eval),1:2]
    best.eval
  }

  split.apply <- function(node)
  {
    tree <<- rbind(tree,
                   data.frame(node=(2*node):(2*node+1), attribute=NA, value=NA, target=NA,
                              count=NA, mean=NA, variance=NA))

    av <- data[[tree$attribute[tree$node==node]]]
    cond <- !is.na(av) & (if (is.numeric(av))
                            av<=as.numeric(tree$value[tree$node==node])
                          else av==tree$value[tree$node==node])
    nodemap[nodemap==node & cond] <<- 2*node
    nodemap[nodemap==node & !cond] <<- 2*node+1
  }

  tree <- nodemap <- node <- NULL
  target <- y.var(formula)
  attributes <- x.vars(formula, data)
  ns <- ifelse(ns==0, round(length(attributes)/3),
               clip.val(ns, 1, length(attributes)))

  init()
  while (is.finite(node))
  {
    target.summary(node)
    target.value(node)
    if (!stop.criteria(node))
      if (split.select(node)<Inf)
        split.apply(node)
    node <- next.node(node)
  }
  `class<-`(tree, "regtree")
}


## generate base models by simple multiple algorithm application
base.ensemble.simple <- function(formula, data, m, alg, args=NULL)
{
  lapply(1:m, function(i) do.call(alg, c(list(formula, data), args)))
}

  # base models for the HouseVotes84 data
hv.bm.tree.rnd <- base.ensemble.simple(Class~., hv.train, 50, grow.randdectree)

  # base models for the BostonHousing data
bh.bm.tree.rnd <- base.ensemble.simple(medv~., bh.train, 50, grow.randregtree,
                                       args=list(minvar=5))

  # base model training set errors for the HouseVotes84 data
hv.train.err.tree.rnd <- sapply(hv.bm.tree.rnd,
                                function(h) err(predict(h, hv.train),
                                                hv.train$Class))

  # base model training set MSE values for the BostonHousing data
bh.train.mse.tree.rnd <- sapply(bh.bm.tree.rnd,
                                function(h) mse(predict(h, bh.train), bh.train$medv))

  # base model test set errors for the HouseVotes84 data
hv.test.err.tree.rnd <- sapply(hv.bm.tree.rnd,
                               function(h) err(predict(h, hv.test), hv.test$Class))

  # base model test set MSE values for the BostonHousing data
bh.test.mse.tree.rnd <- sapply(bh.bm.tree.rnd,
                               function(h) mse(predict(h, bh.test), bh.test$medv))
##############
#15-3-5.R
##############
  # base model training set errors for the HouseVotes84 data
boxplot(list(tree.sx=hv.train.err.tree.sx,
	     tree.wx=hv.train.err.tree.wx,
	     tree.sa=hv.train.err.tree.sa,
	     tree.rnd=hv.train.err.tree.rnd,
	     nb.sx=hv.train.err.nb.sx,
	     nb.sa=hv.train.err.nb.sa),
        main="HouseVotes84 (train)", las=2, col="grey", ylim=c(0, 0.26))

  # base model test set errors for the HouseVotes84 data
boxplot(list(tree.sx=hv.test.err.tree.sx,
	     tree.wx=hv.test.err.tree.wx,
	     tree.sa=hv.test.err.tree.sa,
	     tree.rnd=hv.test.err.tree.rnd,
	     nb.sx=hv.test.err.nb.sx,
	     nb.sa=hv.test.err.nb.sa),
        main="HouseVotes84 (test)", las=2, col="grey", ylim=c(0, 0.26))

  # base model training set MSE values for the BostonHousing data
boxplot(list(tree.sx=bh.train.mse.tree.sx,
	     tree.wx=bh.train.mse.tree.wx,
	     tree.sa=bh.train.mse.tree.sa,
	     tree.rnd=bh.train.mse.tree.rnd,
	     lm.sx=bh.train.mse.lm.sx,
	     lm.wx=bh.train.mse.lm.wx,
             lm.sa=bh.train.mse.lm.sa),
        main="BostonHousing (train)", las=2, col="grey", ylim=c(0, 130))

  # base model test set MSE values for the BostonHousing data
boxplot(list(tree.sx=bh.test.mse.tree.sx,
	     tree.wx=bh.test.mse.tree.wx,
	     tree.sa=bh.test.mse.tree.sa,
	     tree.rnd=bh.test.mse.tree.rnd,
	     lm.sx=bh.test.mse.lm.sx,
	     lm.wx=bh.train.mse.lm.wx,
	     lm.sa=bh.test.mse.lm.sa),
        main="BostonHousing (test)", las=2, col="grey", ylim=c(0, 130))
##############
#15-4-1.R
##############
## combine base models by voting/averaging
predict.ensemble.basic <- function(models, data, predf=predict)
{
  bp <- data.frame(lapply(models, function(h) predf(h, data)))
  combf <- if (is.numeric(bp[,1])) mean else modal  # combination scheme
  cp <- sapply(1:nrow(bp), function(i) combf(as.vector(as.matrix(bp[i,]))))
  if (is.factor(bp[,1]))
    factor(cp, levels=levels(bp[,1]))
  else
    cp
}

  # combine base models for the HouseVotes84 data
hv.pred.tree.sx.b <- predict.ensemble.basic(hv.bm.tree.sx, hv.test,
                                            predf=function(...)
                                                  predict(..., type="c"))
hv.pred.nb.sx.b <- predict.ensemble.basic(hv.bm.nb.sx, hv.test)
hv.pred.tree.wx.b <- predict.ensemble.basic(hv.bm.tree.wx, hv.test,
                                            predf=function(...)
                                                  predict(..., type="c"))
hv.pred.tree.sa.b <- predict.ensemble.basic(hv.bm.tree.sa, hv.test,
                                            predf=function(...)
                                                  predict(..., type="c"))
hv.pred.nb.sa.b <- predict.ensemble.basic(hv.bm.nb.sa, hv.test)
hv.pred.tree.rnd.b <- predict.ensemble.basic(hv.bm.tree.rnd, hv.test)

  # combine base models for the BostonHousing data
bh.pred.tree.sx.b <- predict.ensemble.basic(bh.bm.tree.sx, bh.test)
bh.pred.lm.sx.b <- predict.ensemble.basic(bh.bm.lm.sx, bh.test)
bh.pred.tree.wx.b <- predict.ensemble.basic(bh.bm.tree.wx, bh.test)
bh.pred.lm.wx.b <- predict.ensemble.basic(bh.bm.lm.wx, bh.test)
bh.pred.tree.sa.b <- predict.ensemble.basic(bh.bm.tree.sa, bh.test)
bh.pred.lm.sa.b <- predict.ensemble.basic(bh.bm.lm.sa, bh.test)
bh.pred.tree.rnd.b <- predict.ensemble.basic(bh.bm.tree.rnd, bh.test)

  # ensemble model test set errors for the HouseVotes84 data
hv.err.b <- c(tree = hv.err.tree,
              tree.sx = err(hv.pred.tree.sx.b, hv.test$Class),
              tree.wx = err(hv.pred.tree.wx.b, hv.test$Class),
              tree.sa = err(hv.pred.tree.sa.b, hv.test$Class),
              tree.rnd = err(hv.pred.tree.rnd.b, hv.test$Class),
              nb = hv.err.nb,
              nb.sx = err(hv.pred.nb.sx.b, hv.test$Class),
              nb.sa = err(hv.pred.nb.sa.b, hv.test$Class))

  # ensemble model test set MSE values for the BostonHousing data
bh.mse.b <- c(tree = bh.mse.tree,
              tree.sx = mse(bh.pred.tree.sx.b, bh.test$medv),
              tree.wx = mse(bh.pred.tree.wx.b, bh.test$medv),
              tree.sa = mse(bh.pred.tree.sa.b, bh.test$medv),
              tree.rnd = mse(bh.pred.tree.rnd.b, bh.test$medv),
              lm = bh.mse.lm,
              lm.sx = mse(bh.pred.lm.sx.b, bh.test$medv),
              lm.wx = mse(bh.pred.lm.wx.b, bh.test$medv),
              lm.sa = mse(bh.pred.lm.sa.b, bh.test$medv))

barplot(hv.err.b, main="HouseVotes84", ylab="Error", las=2)
lines(c(0, 10), rep(hv.err.b[1], 2), lty=2)
lines(c(0, 10), rep(hv.err.b[6], 2), lty=3)

barplot(bh.mse.b, main="Boston Housing", ylab="MSE", las=2)
lines(c(0, 11), rep(bh.mse.b[1], 2), lty=2)
lines(c(0, 11), rep(bh.mse.b[6], 2), lty=3)
##############
#15-4-2.R
##############
## combine base models by probability averaging
predict.ensemble.prob <- function(models, data, predf=predict,
                                  prob=FALSE, labels=NULL)
{
  bp <- lapply(models, function(h) predf(h, data))
  cp <- 0
  for (i in (1:(m <- length(bp))))
    cp <- cp + bp[[i]]
  if (prob)
    cp/m
  else
  {
    if (is.null(labels))
      labels <- colnames(cp)
    factor(apply(cp, 1, which.max), levels=1:2, labels=labels)
  }
}

  # combine base models for the HouseVotes84 data
hv.pred.tree.sx.p <- predict.ensemble.prob(hv.bm.tree.sx, hv.test)
hv.pred.nb.sx.p <- predict.ensemble.prob(hv.bm.nb.sx, hv.test,
                                         predf=function(...) predict(..., type="r"))
hv.pred.tree.wx.p <- predict.ensemble.prob(hv.bm.tree.wx, hv.test)
hv.pred.tree.sa.p <- predict.ensemble.prob(hv.bm.tree.sa, hv.test)
hv.pred.nb.sa.p <- predict.ensemble.prob(hv.bm.nb.sa, hv.test,
                                         predf=function(...) predict(..., type="r"))

  # ensemble model test set errors for the HouseVotes84 data
hv.err.p <- c(tree = hv.err.tree,
              tree.sx = err(hv.pred.tree.sx.p, hv.test$Class),
              tree.wx = err(hv.pred.tree.wx.p, hv.test$Class),
              tree.sa = err(hv.pred.tree.sa.p, hv.test$Class),
              nb = hv.err.nb,
              nb.sx = err(hv.pred.nb.sx.p, hv.test$Class),
              nb.sa = err(hv.pred.nb.sa.p, hv.test$Class))

barplot(hv.err.p, main="HouseVotes84", ylab="Error", las=2)
lines(c(0, 9), rep(hv.err.p[1], 2), lty=2)
lines(c(0, 9), rep(hv.err.p[5], 2), lty=3)
##############
#15-4-3.R
##############
## combine base models by weighted voting/averaging/summing
predict.ensemble.weighted <- function(models, weights, data, predf=predict,
                                      summing=FALSE)
{
  bp <- data.frame(lapply(models, function(h) predf(h, data)))
  combf <- if (is.numeric(bp[,1])) weighted.mean
           else weighted.modal  # combination scheme
  cp <- sapply(1:nrow(bp), function(i) combf(as.vector(as.matrix(bp[i,])), weights))
  if (is.numeric(bp[,1]) && summing)
    cp <- cp*sum(weights)  # summing instead of averaging requested
  if (is.factor(bp[,1]))
    factor(cp, levels=levels(bp[,1]))
  else
    cp
}

  # combine base models for the HouseVotes84 data
hv.pred.tree.sx.w <- predict.ensemble.weighted(hv.bm.tree.sx,
                                               1/(hv.train.err.tree.sx+0.01),
                                               hv.test,
                                               predf=function(...)
                                                     predict(..., type="c"))
hv.pred.nb.sx.w <- predict.ensemble.weighted(hv.bm.nb.sx,
                                             1/(hv.train.err.nb.sx+0.01),
                                             hv.test)
hv.pred.tree.wx.w <- predict.ensemble.weighted(hv.bm.tree.wx,
                                               1/(hv.train.err.tree.wx+0.01),
                                               hv.test,
                                               predf=function(...)
                                                     predict(..., type="c"))
hv.pred.tree.sa.w <- predict.ensemble.weighted(hv.bm.tree.sa,
                                               1/(hv.train.err.tree.sa+0.01),
                                               hv.test,
                                               predf=function(...)
                                                     predict(..., type="c"))
hv.pred.nb.sa.w <- predict.ensemble.weighted(hv.bm.nb.sa,
                                             1/(hv.train.err.nb.sa+0.01), hv.test)
hv.pred.tree.rnd.w <- predict.ensemble.weighted(hv.bm.tree.rnd,
                                                1/(hv.train.err.tree.rnd+0.01),
                                                hv.test)

  # combine base models for the BostonHousing data
bh.pred.tree.sx.w <- predict.ensemble.weighted(bh.bm.tree.sx,
                                               1/(bh.train.mse.tree.sx+1), bh.test)
bh.pred.lm.sx.w <- predict.ensemble.weighted(bh.bm.lm.sx,
                                             1/(bh.train.mse.lm.sx+1), bh.test)
bh.pred.tree.wx.w <- predict.ensemble.weighted(bh.bm.tree.wx,
                                               1/(bh.train.mse.tree.wx+1), bh.test)
bh.pred.lm.wx.w <- predict.ensemble.weighted(bh.bm.lm.wx,
                                             1/(bh.train.mse.lm.wx+1), bh.test)
bh.pred.tree.sa.w <- predict.ensemble.weighted(bh.bm.tree.sa,
                                               1/(bh.train.mse.tree.sa+1), bh.test)
bh.pred.lm.sa.w <- predict.ensemble.weighted(bh.bm.lm.sa,
                                             1/(bh.train.mse.lm.sa+1), bh.test)
bh.pred.tree.rnd.w <- predict.ensemble.weighted(bh.bm.tree.rnd,
                                                1/(bh.train.mse.tree.rnd+1), bh.test)

  # ensemble model test set errors for the HouseVotes84 data
hv.err.w <- c(tree = hv.err.tree,
              tree.sx = err(hv.pred.tree.sx.w, hv.test$Class),
              tree.wx = err(hv.pred.tree.wx.w, hv.test$Class),
              tree.sa = err(hv.pred.tree.sa.w, hv.test$Class),
              tree.rnd = err(hv.pred.tree.rnd.w, hv.test$Class),
              nb = hv.err.nb,
              nb.sx = err(hv.pred.nb.sx.w, hv.test$Class),
              nb.sa = err(hv.pred.nb.sa.w, hv.test$Class))

  # ensemble model test set MSE values for the BostonHousing data
bh.mse.w <- c(tree = bh.mse.tree,
              tree.sx = mse(bh.pred.tree.sx.w, bh.test$medv),
              tree.wx = mse(bh.pred.tree.wx.w, bh.test$medv),
              tree.sa = mse(bh.pred.tree.sa.w, bh.test$medv),
              tree.rnd = mse(bh.pred.tree.rnd.w, bh.test$medv),
              lm = bh.mse.lm,
              lm.sx = mse(bh.pred.lm.sx.w, bh.test$medv),
              lm.wx = mse(bh.pred.lm.wx.w, bh.test$medv),
              lm.sa = mse(bh.pred.lm.sa.w, bh.test$medv))

barplot(hv.err.w, main="HouseVotes84", ylab="Error", las=2)
lines(c(0, 10), rep(hv.err.w[1], 2), lty=2)
lines(c(0, 10), rep(hv.err.w[6], 2), lty=3)

barplot(bh.mse.w, main="Boston Housing", ylab="MSE", las=2)
lines(c(0, 11), rep(bh.mse.w[1], 2), lty=2)
lines(c(0, 11), rep(bh.mse.w[6], 2), lty=3)
##############
#15-4-4.R
##############
## combine base models by using as attributes
## create a model using the specified algorithm and training data
combine.ensemble.attributes <- function(models, data, alg, args=NULL, predf=predict,
                                        append=FALSE)
{
  target <- as.character(models[[1]]$terms[[2]])
  tind <- match(target, names(data))
  data.base <- `names<-`(cbind(data.frame(lapply(models,
                                                 function(h) predf(h, data))),
                               data[[target]]),
                         c(paste("h", 1:length(models), sep=""), target))
  if (append)
    data.base <- cbind(data[,-tind], data.base)
  do.call(alg, c(list(make.formula(target, "."), data.base), args))
}

## combine base models by using as attributes
## predict using the specified base models and combined model
predict.ensemble.attributes <- function(combined.model, base.models, data,
                                        combined.predf=predict, base.predf=predict)
{
  data.pred <- `names<-`(data.frame(lapply(base.models,
                                           function(h) base.predf(h, data))),
                         paste("h", 1:length(base.models), sep=""))
  data.pred <- cbind(data, data.pred)  # make the original attributes available
  combined.predf(combined.model, data.pred)
}

  # combine base models for the HouseVotes84 data
hv.tree.sx.nb <- combine.ensemble.attributes(hv.bm.tree.sx, hv.train, naiveBayes,
                                             predf=function(...)
                                                   predict(..., type="c"))
hv.pred.tree.sx.nb <- predict.ensemble.attributes(hv.tree.sx.nb, hv.bm.tree.sx,
                                                  hv.test,
                                                  base.predf=function(...)
                                                             predict(..., type="c"))
hv.tree.wx.nb <- combine.ensemble.attributes(hv.bm.tree.wx, hv.train, naiveBayes,
                                             predf=function(...)
                                                   predict(..., type="c"))
hv.pred.tree.wx.nb <- predict.ensemble.attributes(hv.tree.wx.nb, hv.bm.tree.wx,
                                                  hv.test,
                                                  base.predf=function(...)
                                                             predict(..., type="c"))
hv.tree.sa.nb <- combine.ensemble.attributes(hv.bm.tree.sa, hv.train, naiveBayes,
                                             predf=function(...)
                                                   predict(..., type="c"))
hv.pred.tree.sa.nb <- predict.ensemble.attributes(hv.tree.sa.nb, hv.bm.tree.sa,
                                                  hv.test,
                                                  base.predf=function(...)
                                                             predict(..., type="c"))

  # combine base models for the BostonHousing data
bh.tree.sx.lm <- combine.ensemble.attributes(bh.bm.tree.sx, bh.train, lm)
bh.pred.tree.sx.lm <- predict.ensemble.attributes(bh.tree.sx.lm, bh.bm.tree.sx,
                                                  bh.test)
bh.tree.wx.lm <- combine.ensemble.attributes(bh.bm.tree.wx, bh.train, lm)
bh.pred.tree.wx.lm <- predict.ensemble.attributes(bh.tree.wx.lm, bh.bm.tree.wx,
                                                  bh.test)
bh.tree.sa.lm <- combine.ensemble.attributes(bh.bm.tree.sa, bh.train, lm)
bh.pred.tree.sa.lm <- predict.ensemble.attributes(bh.tree.sa.lm, bh.bm.tree.sa,
                                                  bh.test)

  # ensemble model test set errors for the HouseVotes84 data
hv.err.a <- c(tree = hv.err.tree,
              tree.sx.nb = err(hv.pred.tree.sx.nb, hv.test$Class),
              tree.wx.nb = err(hv.pred.tree.wx.nb, hv.test$Class),
              tree.sa.nb = err(hv.pred.tree.sa.nb, hv.test$Class))

  # ensemble model test set MSE values for the BostonHousing data
bh.mse.a <- c(tree = bh.mse.tree,
              tree.sx.lm = mse(bh.pred.tree.sx.lm, bh.test$medv),
              tree.wx.lm = mse(bh.pred.tree.wx.lm, bh.test$medv),
              tree.sa.lm = mse(bh.pred.tree.sa.lm, bh.test$medv))

barplot(hv.err.a, main="HouseVotes84", ylab="Error", las=2)
lines(c(0, 5), rep(hv.err.a[1], 2), lty=2)

barplot(bh.mse.a, main="Boston Housing", ylab="MSE", las=2)
lines(c(0, 5), rep(bh.mse.a[1], 2), lty=2)
##############
#15-5-1.R
##############
## bagging ensemble modeling using m base models created with algorithm alg
## with arguments arg
bagging <- function(formula, data, m, alg, args=NULL)
{
  `class<-`(base.ensemble.sample.x(formula, data, m, alg, args), "bagging")
}

## bagging prediction
predict.bagging <- function(models, data, predf=predict)
{
  predict.ensemble.basic(models, data, predf)
}

  # bagging for the HouseVotes84 data
hv.bagg.tree <- bagging(Class~., hv.train, 50, rpart, args=list(minsplit=2, cp=0))
hv.bagg.nb <- bagging(Class~., hv.train, 50, naiveBayes)

hv.pred.bagg.tree <- predict(hv.bagg.tree, hv.test,
                             predf=function(...) predict(..., type="c"))
hv.pred.bagg.nb <- predict(hv.bagg.nb, hv.test)

  # bagging for the BostonHousing data
bh.bagg.tree <- bagging(medv~., bh.train, 50, rpart, args=list(minsplit=2, cp=0))
bh.bagg.lm <- bagging(medv~., bh.train, 50, lm)

bh.pred.bagg.tree <- predict(bh.bagg.tree, bh.test)
bh.pred.bagg.lm <- predict(bh.bagg.lm, bh.test)

  # bagging test set errors for the HouseVotes84 data
hv.err.bagg <- list(tree = err(hv.pred.bagg.tree, hv.test$Class),
                    nb = err(hv.pred.bagg.nb, hv.test$Class))

  # bagging test set MSE values for the BostonHousing data
bh.mse.bagg <- list(tree = mse(bh.pred.bagg.tree, bh.test$medv),
                    lm = mse(bh.pred.bagg.lm, bh.test$medv))
##############
#15-5-2.R
##############
curve(0.5*log((1-x)/x), from=0, to=0.5,
      xlab="model error", ylab="model weight")
curve(exp(0.5*log((1-x)/x)), from=0, to=0.5,
      xlab="model error", ylab="instance weight multiplier", ylim=c(0, 10), lty=2)
curve(exp(-0.5*log((1-x)/x)), from=0, to=0.5, lty=3, add=TRUE)
legend("topright", legend=c("misclassified", "correctly classified"), lty=2:3)
##############
#15-5-3.R
##############
## AdaBoost ensemble modeling using up to m base models created using algorithm alg
## with arguments arg and maximum allowed base model error 0.5-eps
adaboost <- function(formula, data, m, alg, eps=0.01, args=NULL, predf=predict)
{
  class <- y.var(formula)
  nc <- nlevels(data[[class]])
  model.weights <- NULL

  abst.reweight <- function(weights, pred)
  {
    e <- werr(pred, data[[class]], weights)
    if (e<=0.5-eps && is.finite(mw <- 0.5*(log((1-e)/e)+log(nc-1))))
    {
      model.weights <<- c(model.weights, mw)
      weights*exp(mw*(2*(pred!=data[[class]])-1))
    }
    else
      NULL
  }

  `class<-`(list(models=base.ensemble.weight.x(formula, data, m, alg, args,
                                               weights=rep(1, nrow(data)),
                                               abst.reweight,
                                               predf=predf),
                 model.weights=model.weights), "adaboost")
}

## AdaBoost prediction
predict.adaboost <- function(boost, data, predf=predict)
{
  predict.ensemble.weighted(boost$models, boost$model.weights, data, predf)
}

  # AdaBoost for the HouseVotes84 data
hv.abst.tree1 <- adaboost(Class~., hv.train, 50, rpart,
                          args=list(minsplit=2, cp=0, maxdepth=1),
                          predf=function(...) predict(..., type="c"))
hv.abst.tree3 <- adaboost(Class~., hv.train, 50, rpart,
                          args=list(minsplit=2, cp=0, maxdepth=3),
                          predf=function(...) predict(..., type="c"))
hv.abst.tree5 <- adaboost(Class~., hv.train, 50, rpart,
                          args=list(minsplit=2, cp=0, maxdepth=5),
                          predf=function(...) predict(..., type="c"))

hv.pred.abst.tree1 <- predict(hv.abst.tree1, hv.test,
                              predf=function(...) predict(..., type="c"))
hv.pred.abst.tree3 <- predict(hv.abst.tree3, hv.test,
                              predf=function(...) predict(..., type="c"))
hv.pred.abst.tree5 <- predict(hv.abst.tree5, hv.test,
                              predf=function(...) predict(..., type="c"))

  # AdaBoost test set errors for the HouseVotes84 data
hv.err.abst <- list(tree1 = err(hv.pred.abst.tree1, hv.test$Class),
                    tree3 = err(hv.pred.abst.tree3, hv.test$Class),
                    tree5 = err(hv.pred.abst.tree5, hv.test$Class))
##############
#15-5-4.R
##############
## gradient boosting ensemble modeling using up to m base models
## created with algorithm alg with arguments arg
gradboost <- function(formula, data, m, alg, beta=0.1, args=NULL, predf=predict)
{
  attributes <- x.vars(formula, data)
  aind <- match(attributes, names(data))
  f <- y.var(formula)
  find <- match(f, names(data))

  models <- list(do.call(alg, c(list(formula, data), args)))
  model.weights <- 1

  for (i in (2:m))
  {
    res <- data[,find]-predict.gradboost(list(models=models,
                                              model.weights=model.weights),
                                         data, predf=predf)
    data.i <- eval(parse(text=paste("cbind(data[,aind],", f, "=res)")))
    models <- c(models, list(h <- do.call(alg, c(list(formula, data.i), args))))
    model.weights <- c(model.weights,
                       beta*sum(res*(pred <- predf(h, data)))/sum(pred^2))
  }
  `class<-`(list(models=models, model.weights=model.weights), "gradboost")
}

## gradient boosting prediction
predict.gradboost <- function(boost, data, predf=predict)
{
  predict.ensemble.weighted(boost$models, boost$model.weights, data, predf,
                            summing=TRUE)
}

  # gradient boosting for the BostonHousing data
bh.gbst.tree1 <- gradboost(medv~., bh.train, 50, rpart,
                           args=list(minsplit=2, cp=0, maxdepth=1))
bh.gbst.tree3 <- gradboost(medv~., bh.train, 50, rpart,
                           args=list(minsplit=2, cp=0, maxdepth=3))
bh.gbst.tree5 <- gradboost(medv~., bh.train, 50, rpart,
                           args=list(minsplit=2, cp=0, maxdepth=5))
bh.gbst.lm <- gradboost(medv~., bh.train, 50, lm)

bh.pred.gbst.tree1 <- predict(bh.gbst.tree1, bh.test)
bh.pred.gbst.tree3 <- predict(bh.gbst.tree3, bh.test)
bh.pred.gbst.tree5 <- predict(bh.gbst.tree5, bh.test)
bh.pred.gbst.lm <- predict(bh.gbst.lm, bh.test)

  # gradient boosting test set MSE values for the BostonHousing data
bh.mse.gbst <- list(tree1 = mse(bh.pred.gbst.tree1, bh.test$medv),
                    tree3 = mse(bh.pred.gbst.tree3, bh.test$medv),
                    tree5 = mse(bh.pred.gbst.tree5, bh.test$medv),
                    lm = mse(bh.pred.gbst.lm, bh.test$medv))
##############
#15-5-5.R
##############
## random forest ensemble modeling using m randomized decision or regression trees
## with ns randomly selected attributes at each node used for splitting
randforest <- function(formula, data, m, ns=0, args=NULL)
{
  target <- y.var(formula)
  alg <- if (!is.numeric(data[[target]])) grow.randdectree else grow.randregtree

  `class<-`(base.ensemble.sample.x(formula, data, m, alg, args=c(list(ns=ns), args)),
            "randforest")
}

## random forest prediction
predict.randforest <- function(rf, data)
{
  predict.ensemble.basic(rf, data)
}

  # random forest for the HouseVotes84 data
hv.rf.tree3 <- randforest(Class~., hv.train, 50, args=list(maxdepth=3))
hv.rf.tree5 <- randforest(Class~., hv.train, 50, args=list(maxdepth=5))
hv.rf.tree8 <- randforest(Class~., hv.train, 50, args=list(maxdepth=8))

hv.pred.rf.tree3 <- predict(hv.rf.tree3, hv.test)
hv.pred.rf.tree5 <- predict(hv.rf.tree5, hv.test)
hv.pred.rf.tree8 <- predict(hv.rf.tree8, hv.test)

  # random forest for the BostonHousing data
bh.rf.tree3 <- randforest(medv~., bh.train, 50, args=list(maxdepth=3))
bh.rf.tree5 <- randforest(medv~., bh.train, 50, args=list(maxdepth=5))
bh.rf.tree8 <- randforest(medv~., bh.train, 50, args=list(maxdepth=8))

bh.pred.rf.tree3 <- predict(bh.rf.tree3, bh.test)
bh.pred.rf.tree5 <- predict(bh.rf.tree8, bh.test)
bh.pred.rf.tree8 <- predict(bh.rf.tree8, bh.test)

  # random forest test set errors for the HouseVotes84 data
hv.err.rf <- list(tree3 = err(hv.pred.rf.tree3, hv.test$Class),
                  tree5 = err(hv.pred.rf.tree5, hv.test$Class),
                  tree8 = err(hv.pred.rf.tree8, hv.test$Class))

  # random forest test set MSE values for the BostonHousing data
bh.mse.rf <- list(tree3 = mse(bh.pred.rf.tree3, bh.test$medv),
                  tree5 = mse(bh.pred.rf.tree5, bh.test$medv),
                  tree8 = mse(bh.pred.rf.tree8, bh.test$medv))
##############
#15-5-6.R
##############
## random naive Bayes ensemble modeling using m base models
## each with ns randomly selected attributes
## (if unspecified, it defaults to the square root of the number of attributes)
randnaiveBayes <- function(formula, data, m, ns=0)
{
  attributes <- x.vars(formula, data)
  target <- y.var(formula)
  ns <- ifelse(ns==0, round(sqrt(length(attributes))),
                      clip.val(ns, 1, length(attributes)))

  `class<-`(lapply(1:m, function(i)
                        {
                          bag <- sample(nrow(data), size=nrow(data), replace=TRUE)
                          sa <- sample(length(attributes), ns)
                          naiveBayes(make.formula(target, attributes[sa]),
                                     data[bag,])
                        }), "randnaiveBayes")
}

## random naive Bayes prediction
predict.randnaiveBayes <- function(rnb, data, prob=FALSE)
{
  predict.ensemble.prob(rnb, data, predf=function(...) predict(..., type="r"),
                        prob=prob, labels=rnb[[1]]$levels)
}

  # random naive Bayes for the HouseVotes84 data
hv.rnb <- randnaiveBayes(Class~., hv.train, 500)
hv.pred.rnb <- predict(hv.rnb, hv.test)
  # random naive Bayes test set error for the HouseVotes84 data
hv.err.rnb <- list(nb = err(hv.pred.rnb, hv.test$Class))
##############
#15-6-1.R
##############
hv.err <- c(tree=hv.err.tree,
            bagg.tree=hv.err.bagg$tree,
            abst.tree1=hv.err.abst$tree1,
            abst.tree3=hv.err.abst$tree3,
            abst.tree5=hv.err.abst$tree5,
            rf.tree3=hv.err.rf$tree3,
            rf.tree5=hv.err.rf$tree5,
            rf.tree8=hv.err.rf$tree8,
            nb=hv.err.nb,
            bagg.nb=hv.err.bagg$nb,
            rnb=hv.err.rnb$nb)

bh.mse <- c(tree=bh.mse.tree,
            bagg.tree=bh.mse.bagg$tree,
            gbst.tree1=bh.mse.gbst$tree1,
            gbst.tree3=bh.mse.gbst$tree3,
            gbst.tree5=bh.mse.gbst$tree5,
            rf.tree3=bh.mse.rf$tree3,
            rf.tree5=bh.mse.rf$tree5,
            rf.tree8=bh.mse.rf$tree8,
            lm=bh.mse.lm,
            bagg.lm=bh.mse.bagg$lm,
            gbst.lm=bh.mse.gbst$lm)

barplot(hv.err, main="HouseVotes84", ylab="Error", las=2)
lines(c(0, 13), rep(hv.err[1], 2), lty=2)
lines(c(0, 13), rep(hv.err[9], 2), lty=3)

barplot(bh.mse, main="Boston Housing", ylab="MSE", las=2)
lines(c(0, 13), rep(bh.mse[1], 2), lty=2)
lines(c(0, 13), rep(bh.mse[9], 2), lty=3)
##############
#16-1-1.R
##############
library(dmr.claseval)
library(dmr.linclas)
library(dmr.regeval)
library(dmr.trans)
library(dmr.util)

library(lattice)
library(quadprog)
library(kernlab)
library(Matrix)

data(PimaIndiansDiabetes, package="mlbench")
data(BostonHousing, package="mlbench")

set.seed(12)

rpid <- runif(nrow(PimaIndiansDiabetes))
pid.train <- PimaIndiansDiabetes[rpid>=0.33,]
pid.test <- PimaIndiansDiabetes[rpid<0.33,]

rbh <- runif(nrow(BostonHousing))
bh.train <- BostonHousing[rbh>=0.33,-4]
bh.test <- BostonHousing[rbh<0.33,-4]

pid.stdm <- std.all(diabetes~., pid.train)
pid.std.train <- predict.std(pid.stdm, pid.train)
pid.std.test <- predict.std(pid.stdm, pid.test)

bh.stdm <- std.all(medv~., bh.train)
bh.std.train <- predict.std(bh.stdm, bh.train)
bh.std.test <- predict.std(bh.stdm, bh.test)

set.seed(12)

  # dataset for plots
kmf.plot <- function(a1, a2) { 2*a1-3*a2+4 }
kmdat.plot <- `names<-`(expand.grid(seq(1, 5, 0.05), seq(1, 5, 0.05)), c("a1", "a2"))
kmdat.plot$f <- kmf.plot(kmdat.plot$a1, kmdat.plot$a2)
kmdat.plot$c <- as.factor(ustep(kmdat.plot$f))

  # datasets for parameter estimation examples
kmg <- function(a1, a2, a3, a4) { a1^2+2*a2^2-a3^2-2*a4^2+2*a1-3*a2+2*a3-3*a4+1 }
kmf <- function(a1, a2, a3, a4) { 3*a1+4*a2-2*a3+2*a4-3 }
kmdat <- data.frame(a1=runif(400, min=1, max=5), a2=runif(400, min=1, max=5),
                    a3=runif(400, min=1, max=5), a4=runif(400, min=1, max=5))
kmdat$g <- kmg(kmdat$a1, kmdat$a2, kmdat$a3, kmdat$a4)
kmdat$c <- as.factor(ustep(kmdat$g))
kmdat$f <- kmf(kmdat$a1, kmdat$a2, kmdat$a3, kmdat$a4)

kmdat.train <- kmdat[1:200,]
kmdat.test <- kmdat[201:400,]

  # linearly separable training and test subsets
kmdat.ls <- linsep.sub(c~a1+a2+a3+a4, kmdat)
kmdat.train.ls <- kmdat[1:200,][kmdat.ls[1:200],]
kmdat.test.ls <- kmdat[201:400,][kmdat.ls[201:400],]
##############
#16-2-1.R
##############
## functional margin of w with respect to instances from data
## using the cvec vector of {-1, 1} class labels
fmarg <- function(w, data, cvec)
{ cvec*predict.par(list(repf=repf.linear, w=w), data) }

## geometric margin of w with respect to instances from data
## using the cvec vector of {-1, 1} class labels
gmarg <- function(w, data, cvec) { fmarg(w, data, cvec)/l2norm(w[-length(w)]) }

## plot separating and b-margin lines for linear threshold classification
## with 2 attributes
plot.margin <- function(w, data, cvec, b=1, add=FALSE,
                        col.sep="black", col.pos="grey70", col.neg="grey30", ...)
{
  # y value corresponding to x on the regression line represented by w
  lry <- function(x, w) {sum(-w[c(1,3)]/w[2]*c(x, 1)) }

  if (!add)
  {
    plot(data[,1][cvec==1], data[,2][cvec==1], col=col.pos,
       xlab="a1", ylab="a2", xlim=range(data[,1]), ylim=range(data[,2]), ...)
    points(data[,1][cvec!=1], data[,2][cvec!=1], col=col.neg, ...)
  }

  lines(range(data[,1]), c(lry(min(data[,1]), w),
                           lry(max(data[,1]), w)), col=col.sep, ...)
  lines(range(data[,1]), c(lry(min(data[,1]), w-c(0, 0, b)),
                           lry(max(data[,1]), w-c(0, 0, b))), col=col.pos, ...)
  lines(range(data[,1]), c(lry(min(data[,1]), w+c(0, 0, b)),
                           lry(max(data[,1]), w+c(0, 0, b))), col=col.neg, ...)
  list(fmargin=min(fmarg(w, data, cvec)), gmargin=min(gmarg(w, data, cvec)))
}

  # dataset for margin illustration (skip near-boundary instances from kmdat.plot)
kmdat.m <- kmdat.plot[abs(kmdat.plot$f)>2,c("a1", "a2", "c")]
kmdat.m <- kmdat.m[sample(nrow(kmdat.m), 100),]

  # parameter vector for margin demonstration
w.m <- c(1, -2)
  # predictions with intercept 0
p0.m <- predict.par(list(repf=repf.linear, w=c(w.m, 0)), kmdat.m[,1:2])
 # symmetric-margin intercept
w.m <- c(w.m, -(max(p0.m[kmdat.m$c==0])+min(p0.m[kmdat.m$c==1]))/2)

  # minimum functional margin
min(fmarg(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1))
  # minimum geometric
min(gmarg(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1))

  # scale parameters to get minimum functional margin of 1
w.m <- w.m/min(fmarg(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1))
  # minimum functional margin after parameter scaling (1)
min(fmarg(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1))
  # minimum geometric margin after parameter scaling (unchanged)
min(gmarg(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1))

plot.margin(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1)
##############
#16-2-2.R
##############
## linear SVM parameter estimation using primal-form quadratic programming
## solvers: "solve.QP" or "ipop"
svm.linear.prim <- function(formula, data, svthres=1e-9, inf=1e3, solver="solve.QP")
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  cvec <- 2*as.num0(data[[class]])-1  # class vector using {-1, 1} labels
  amat <- cbind(as.matrix(data[,aind]), intercept=1)  # attribute value matrix

  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(rbind(cbind(diag(sum(aind)), 0), 0))$mat,
                 dvec=rep(0, sum(aind)+1),
                 Amat=t(rmm(amat, cvec)),
                 bvec=rep(1, nrow(data)))
  else if (solver=="ipop")
    args <- list(c=rep(0, sum(aind)+1),
                 H=rbind(cbind(diag(sum(aind)), 0), 0),
                 A=rmm(amat, cvec),
                 b=rep(1, nrow(data)),
                 l=rep(-inf, sum(aind)+1),
                 u=rep(inf, sum(aind)+1),
                 r=rep(inf, nrow(data)))
  else stop("Unknown solver: ", solver)

  qp <- do.call(solver, args)
  w <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  sv <- unname(which(cvec*predict.par(list(repf=repf.linear, w=w),
                                      data[,aind,drop=FALSE])<=1+svthres))
  list(model=`class<-`(list(repf=repf.threshold(repf.linear), w=w), "par"), sv=sv)
}

  # estimate linear SVM model parameters
svm.p.ls <- svm.linear.prim(c~a1+a2+a3+a4, kmdat.train.ls)

  # misclassification error
err(predict(svm.p.ls$model, kmdat.train.ls[,1:4]), kmdat.train.ls$c)
err(predict(svm.p.ls$model, kmdat.test.ls[,1:4]), kmdat.test.ls$c)
##############
#16-2-3.R
##############
  # hard-margin SVM
svm.mh <- svm.linear.prim(c~., kmdat.m, solver="ipop")

  # optimal separating and margin lines
plot.margin(svm.mh$model$w, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1)

  # suboptimal separating and margin lines for comparison
plot.margin(w.m, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1, add=TRUE, lty=3)
##############
#16-2-4.R
##############
## linear SVM parameter estimation using dual-form quadratic programming
## solvers: "solve.QP" or "ipop"
svm.linear.dual <- function(formula, data, svthres=1e-3, inf=1e3, solver="solve.QP")
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  cvec <- 2*as.num0(data[[class]])-1  # class vector using {-1, 1} labels
  ccmat <- outer(cvec, cvec)          # class-class product matrix
  amat <- as.matrix(data[,aind])      # attribute value matrix
  dpmat <- amat%*%t(amat)             # dot product matrix

  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(dpmat*ccmat)$mat,
                 dvec=rep(1, nrow(data)),
                 Amat=matrix(c(cvec, diag(1, nrow(data))), nrow=nrow(data)),
                 bvec=rep(0, nrow(data)+1),
                 meq=1)
  else if (solver=="ipop")
    args <- list(c=rep(-1, nrow(data)),
                 H=dpmat*ccmat,
                 A=cvec,
                 b=0,
                 l=rep(0, nrow(data)),
                 u=rep(inf, nrow(data)),
                 r=0)
  else
    stop("Unknown solver: ", solver)

  qp <- do.call(solver, args)
  alpha <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  sv <- which(alpha>svthres)
  w <- c(colSums(rmm(amat[sv,], cvec[sv]*alpha[sv])))  # no intercept yet
  p0 <- predict.par(list(repf=repf.linear, w=c(w, 0)), data[,aind,drop=FALSE])
  w <- c(w, intercept=-(max(p0[cvec==-1])+min(p0[cvec==1]))/2)
  list(model=`class<-`(list(repf=repf.threshold(repf.linear), w=w), "par"), sv=sv)
}

  # estimate linear SVM model parameters
svm.d.ls <- svm.linear.dual(c~a1+a2+a3+a4, kmdat.train.ls)

  # misclassification error
err(predict(svm.d.ls$model, kmdat.train.ls[,1:4]), kmdat.train.ls$c)
err(predict(svm.d.ls$model, kmdat.test.ls[,1:4]), kmdat.test.ls$c)
##############
#16-2-5.R
##############
## linear soft-margin SVM parameter estimation using quadratic programming
## solvers: "solve.QP" or "ipop"
svm.linear <- function(formula, data, cost=1, svthres=1e-3, solver="solve.QP")
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  cvec <- 2*as.num0(data[[class]])-1  # class vector using {-1, 1} labels
  ccmat <- outer(cvec, cvec)          # class-class product matrix
  amat <- as.matrix(data[,aind])      # attribute value matrix
  dpmat <- amat%*%t(amat)             # dot product matrix

  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(dpmat*ccmat)$mat,
                 dvec=rep(1, nrow(data)),
                 Amat=matrix(c(cvec, diag(1, nrow(data)), diag(-1, nrow(data))),
                             nrow=nrow(data)),
                 bvec=c(0, rep(0, nrow(data)), rep(-cost, nrow(data))),
                 meq=1)
  else if (solver=="ipop")
    args <- list(c=rep(-1, nrow(data)),
                 H=dpmat*ccmat,
                 A=cvec,
                 b=0,
                 l=rep(0, nrow(data)),
                 u=rep(cost, nrow(data)),
                 r=0)
  else
    stop("Unknown solver: ", solver)

  qp <- do.call(solver, args)
  alpha <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  sv <- which(alpha>svthres)
  w <- c(colSums(rmm(amat[sv,], cvec[sv]*alpha[sv])))  # no intercept yet
  i <- which.min(abs(alpha-cost/2))
  w <- c(w, intercept=cvec[i]-unname(predict.par(list(repf=repf.linear, w=c(w, 0)),
                                                 data[i,aind,drop=FALSE])))
  list(model=`class<-`(list(repf=repf.threshold(repf.linear), w=w), "par"), sv=sv)
}

  # linear SVM for the artificial data
svm.s <- svm.linear(c~a1+a2+a3+a4, kmdat.train)
svm.s.ls <- svm.linear(c~a1+a2+a3+a4, kmdat.train.ls)

svm.s.01 <- svm.linear(c~a1+a2+a3+a4, kmdat.train, cost=0.1)
svm.s.ls.01 <- svm.linear(c~a1+a2+a3+a4, kmdat.train.ls, cost=0.1)

svm.s.10 <- svm.linear(c~a1+a2+a3+a4, kmdat.train, cost=10)
svm.s.ls.10 <- svm.linear(c~a1+a2+a3+a4, kmdat.train.ls, cost=10)

  # linear SVM for the Pima Indians Diabetes data
pid.svm.s <- svm.linear(diabetes~., pid.std.train)
pid.svm.s.01 <- svm.linear(diabetes~., pid.std.train, cost=0.1)
pid.svm.s.10 <- svm.linear(diabetes~., pid.std.train, cost=10)

  # training set misclassification error
err(predict(svm.s$model, kmdat.train[,1:4]), kmdat.train$c)
err(predict(svm.s.01$model, kmdat.train[,1:4]), kmdat.train$c)
err(predict(svm.s.10$model, kmdat.train[,1:4]), kmdat.train$c)

err(predict(svm.s.ls$model, kmdat.train.ls[,1:4]), kmdat.train.ls$c)
err(predict(svm.s.ls.01$model, kmdat.train.ls[,1:4]), kmdat.train.ls$c)
err(predict(svm.s.ls.10$model, kmdat.train.ls[,1:4]), kmdat.train.ls$c)

err(factor(predict(pid.svm.s$model, pid.std.train[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.std.train$diabetes)
err(factor(predict(pid.svm.s.01$model, pid.std.train[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.std.train$diabetes)
err(factor(predict(pid.svm.s.10$model, pid.std.train[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.std.train$diabetes)

  # test set misclassification error
err(predict(svm.s$model, kmdat.test[,1:4]), kmdat.test$c)
err(predict(svm.s.01$model, kmdat.test[,1:4]), kmdat.test$c)
err(predict(svm.s.10$model, kmdat.test[,1:4]), kmdat.test$c)

err(predict(svm.s.ls$model, kmdat.test.ls[,1:4]), kmdat.test.ls$c)
err(predict(svm.s.ls.01$model, kmdat.test.ls[,1:4]), kmdat.test.ls$c)
err(predict(svm.s.ls.10$model, kmdat.test.ls[,1:4]), kmdat.test.ls$c)

err(factor(predict(pid.svm.s$model, pid.std.test[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.test$diabetes)
err(factor(predict(pid.svm.s.01$model, pid.std.test[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.test$diabetes)
err(factor(predict(pid.svm.s.10$model, pid.std.test[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.test$diabetes)
##############
#16-2-6.R
##############
  # soft-margin SVM
svm.ms.1 <- svm.linear(c~., kmdat.m, solver="ipop", cost=1)
w.ms.1 <- svm.ms.1$model$w

svm.ms.01 <- svm.linear(c~., kmdat.m, solver="ipop", cost=0.1)
w.ms.01 <- svm.ms.01$model$w

  # soft margin: geometric margin corresponding to functional margin of 1
#1/l2norm(w.ms.1[-length(w.ms.1)])
#1/l2norm(w.ms.01[-length(w.ms.01)])

  # separating and margin lines for cost=1
plot.margin(w.ms.1, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1, main="Linearly separable")
  # separating and margin lines for cost=0.1
plot.margin(w.ms.01, kmdat.m[,1:2], 2*as.num0(kmdat.m$c)-1, add=TRUE, lty=3)

  # the same for linearly inseparable data

kmdat.m.nls <- kmdat.m
kmdat.m.nls$c <- as.factor(ifelse(runif(nrow(kmdat.m))<0.1,
                                  1-as.numchar(kmdat.m$c), as.numchar(kmdat.m$c)))

svm.ms.nls.1 <- svm.linear(c~., kmdat.m.nls, solver="ipop", cost=1)
w.ms.nls.1 <- svm.ms.nls.1$model$w

svm.ms.nls.01 <- svm.linear(c~., kmdat.m.nls, solver="ipop", cost=0.1)
w.ms.nls.01 <- svm.ms.nls.01$model$w

  # soft margin: geometric margin corresponding to functional margin of 1
#1/l2norm(w.ms.nls.1[-length(w.ms.nls.1)])
#1/l2norm(w.ms.nls.01[-length(w.ms.nls.01)])

  # separating and margin lines for cost=1
plot.margin(w.ms.nls.1, kmdat.m.nls[,1:2], 2*as.num0(kmdat.m.nls$c)-1,
            main="Linearly inseparable")
  # separating and margin lines for cost=0.1
plot.margin(w.ms.nls.01, kmdat.m.nls[,1:2], 2*as.num0(kmdat.m.nls$c)-1,
            add=TRUE, lty=3)
##############
#16-3-1.R
##############
## plot regression tube lines for linear regression
## with a single attributes
plot.tube <- function(w, data, eps, add=FALSE,
                      col.point="black", col.line="black", ...)
{
  # y value corresponding to x on the regression line represented by w
  lry <- function(x, w) {sum(w*c(x, 1)) }

  if (!add)
    plot(data[,1], data[,2], col=col.point,
         xlab="a1", ylab="h", xlim=range(data[,1]), ylim=range(data[,2]), ...)

  lines(range(data[,1]), c(lry(min(data[,1]), w), lry(max(data[,1]), w)),
        col=col.line)
  lines(range(data[,1]), c(lry(min(data[,1]), w-c(0, eps)),
                           lry(max(data[,1]), w-c(0, eps))), col=col.line, lty=3)
  lines(range(data[,1]), c(lry(min(data[,1]), w+c(0, eps)),
                           lry(max(data[,1]), w+c(0, eps))), col=col.line, lty=3)
}

  # dataset for tube demonstration (take instances with similar a2 values)
kmdat.t <- kmdat.plot[abs(kmdat.plot$a2-mean(kmdat.plot$a2))<1,]
kmdat.t <- kmdat.t[sample(nrow(kmdat.t), 100), c("a1", "f")]

  # parameter vector for tube demonstration
w.t <- lm(f~a1, kmdat.t)$coef[2:1]

plot.tube(w.t, kmdat.t, eps=1)
##############
#16-3-2.R
##############
## linear SVR parameter estimation using quadratic programming
## solvers: "solve.QP" or "ipop"
svr.linear <- function(formula, data, eps=0.01, cost=1, svthres=1e-3,
                       solver="solve.QP")
{
  f <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  fvec <- data[[f]]                   # target function vector
  amat <- as.matrix(data[,aind])      # attribute value matrix
  dpmat <- amat%*%t(amat)             # dot product matrix

  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(rbind(cbind(dpmat, -dpmat), cbind(-dpmat, dpmat)))$mat,
                 dvec=c(fvec-eps, -fvec-eps),
                 Amat=matrix(c(rep(1, nrow(data)), rep(-1, nrow(data)),
                               diag(1, 2*nrow(data)), diag(-1, 2*nrow(data))),
                             nrow=2*nrow(data)),
                 bvec=c(0, rep(0, 2*nrow(data)), rep(-cost, 2*nrow(data))),
                 meq=1)
  else if (solver=="ipop")
    args <- list(c=c(-fvec+eps, fvec+eps),
                 H=rbind(cbind(dpmat, -dpmat), cbind(-dpmat, dpmat)),
                 A=c(rep(1, nrow(data)), rep(-1, nrow(data))),
                 b=0,
                 l=rep(0, 2*nrow(data)),
                 u=rep(cost, 2*nrow(data)),
                 r=0)
  else
    stop("Unknown solver: ", solver)

  qp <- do.call(solver, args)
  alpha <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  beta <- alpha[1:nrow(data)]-alpha[(nrow(data)+1):(2*nrow(data))]
  sv <- which(abs(beta)>svthres)
  w <- c(colSums(rmm(amat[sv,], beta[sv])))  # no intercept yet
  i <- which.min(abs(beta-cost/2))
  w <- c(w, intercept=fvec[i]-unname(predict.par(list(repf=repf.linear, w=c(w, 0)),
                                                 data[i,aind,drop=FALSE]))-
                        sign(beta[i])*eps)
  list(model=`class<-`(list(repf=repf.linear, w=w), "par"), sv=sv)
}

  # linear SVR for f
svrf <- svr.linear(f~a1+a2+a3+a4, kmdat.train)
svrf.e1 <- svr.linear(f~a1+a2+a3+a4, eps=1, kmdat.train)
svrf.c01 <- svr.linear(f~a1+a2+a3+a4, cost=0.1, kmdat.train)

  # linear SVR for g
svrg <- svr.linear(g~a1+a2+a3+a4, kmdat.train)
svrg.e1 <- svr.linear(g~a1+a2+a3+a4, eps=1, kmdat.train)
svrg.c01 <- svr.linear(g~a1+a2+a3+a4, cost=0.1, kmdat.train)

  # linear SVR for the Boston Housing data
bh.svr <- svr.linear(medv~., bh.std.train)
bh.svr.e1 <- svr.linear(medv~., eps=1, bh.std.train)
bh.svr.c01 <- svr.linear(medv~., cost=0.1, bh.std.train)

  # training set MSE
mse(predict(svrf$model, kmdat.train[,1:4]), kmdat.train$f)
mse(predict(svrf.e1$model, kmdat.train[,1:4]), kmdat.train$f)
mse(predict(svrf.c01$model, kmdat.train[,1:4]), kmdat.train$f)

mse(predict(svrg$model, kmdat.train[,1:4]), kmdat.train$g)
mse(predict(svrg.e1$model, kmdat.train[,1:4]), kmdat.train$g)
mse(predict(svrg.c01$model, kmdat.train[,1:4]), kmdat.train$g)

mse(predict(bh.svr$model, bh.std.train[,-13]), bh.std.train$medv)
mse(predict(bh.svr.e1$model, bh.std.train[,-13]), bh.std.train$medv)
mse(predict(bh.svr.c01$model, bh.std.train[,-13]), bh.std.train$medv)

  # test set MSE
mse(predict(svrf$model, kmdat.test[,1:4]), kmdat.test$f)
mse(predict(svrf.e1$model, kmdat.test[,1:4]), kmdat.test$f)
mse(predict(svrf.c01$model, kmdat.test[,1:4]), kmdat.test$f)

mse(predict(svrg$model, kmdat.test[,1:4]), kmdat.test$g)
mse(predict(svrg.e1$model, kmdat.test[,1:4]), kmdat.test$g)
mse(predict(svrg.c01$model, kmdat.test[,1:4]), kmdat.test$g)

mse(predict(bh.svr$model, bh.std.test[,-13]), bh.std.test$medv)
mse(predict(bh.svr.e1$model, bh.std.test[,-13]), bh.std.test$medv)
mse(predict(bh.svr.c01$model, bh.std.test[,-13]), bh.std.test$medv)
##############
#16-3-3.R
##############
  # eps=0.5, cost=1
svr.t.05.1 <- svr.linear(f~., kmdat.t, solver="ipop", eps=0.5, cost=1)
w.t.05.1 <- svr.t.05.1$model$w
l2norm(w.t.05.1[-length(w.t.05.1)])

  # eps=0.5, cost=0.1
svr.t.05.01 <- svr.linear(f~., kmdat.t, solver="ipop", eps=0.5, cost=0.1)
w.t.05.01 <- svr.t.05.01$model$w
l2norm(w.t.05.01[-length(w.t.05.01)])

  # eps=0.5, cost=0.01
svr.t.05.001 <- svr.linear(f~., kmdat.t, solver="ipop", eps=0.5, cost=0.01)
w.t.05.001 <- svr.t.05.001$model$w
l2norm(w.t.05.001[-length(w.t.05.001)])

  # eps=1, cost=1
svr.t.1.1 <- svr.linear(f~., kmdat.t, solver="ipop", eps=1, cost=1)
w.t.1.1 <- svr.t.1.1$model$w
l2norm(w.t.1.1[-length(w.t.1.1)])

  # eps=1, cost=0.1
svr.t.1.01 <- svr.linear(f~., kmdat.t, solver="ipop", eps=1, cost=0.1)
w.t.1.01 <- svr.t.1.01$model$w
l2norm(w.t.1.01[-length(w.t.1.01)])

  # eps=1, cost=0.01
svr.t.1.001 <- svr.linear(f~., kmdat.t, solver="ipop", eps=1, cost=0.01)
w.t.1.001 <- svr.t.1.001$model$w
l2norm(w.t.1.001[-length(w.t.1.001)])

par(mfcol=c(3, 2))

plot.tube(w.t.05.1, kmdat.t, eps=0.5, main="eps=0.5, cost=1")
plot.tube(w.t.05.01, kmdat.t, eps=0.5, main="eps=0.5, cost=0.1")
plot.tube(w.t.05.001, kmdat.t, eps=0.5, main="eps=0.5, cost=0.01")

plot.tube(w.t.1.1, kmdat.t, eps=1, main="eps=1, cost=1")
plot.tube(w.t.1.01, kmdat.t, eps=1, main="eps=1, cost=0.1")
plot.tube(w.t.1.001, kmdat.t, eps=1, main="eps=1, cost=0.01")
##############
#16-4-1.R
##############
## data transformation that generates new attributes
## defined as the products of all original attribute pairs
trans.mult2 <- function(data)
{
  t(apply(data, 1, function(d) d %o% d))
}

  # original dataset
kmdat.orig <- kmdat.train[1:10,1:4]
  # dot product matrix for the original dataset
kmdat.dp <- as.matrix(kmdat.orig) %*% t(kmdat.orig)

  # transformed dataset
kmdat.trans <- trans.mult2(kmdat.orig)
  # dot product matrix for the transformed dataset
kmdat.dpt <- kmdat.trans %*% t(kmdat.trans)

  # verify that the dot product matrix for the transformed dataset
  # is the same as the squared original dot product matrix
max(abs((kmdat.dpt-kmdat.dp^2)))
##############
#16-5-1.R
##############
## can be called for both single attribute value vectors and for the whole dataset
kernel.linear <- function(av1, av2=av1) { as.matrix(av1)%*%t(av2) }

## can be called for both single attribute value vectors and for the whole dataset
kernel.polynomial <- function(av1, av2=av1, gamma=1, b=0, p=3)
{ (gamma*(as.matrix(av1)%*%t(av2))+b)^p }

## can be called for both single attribute value vectors and for the whole dataset
kernel.radial <- function(av1, av2=av1, gamma=1)
{
  exp(-gamma*outer(1:nrow(av1 <- as.matrix(av1)), 1:ncol(av2 <- t(av2)),
                   Vectorize(function(i, j) l2norm(av1[i,]-av2[,j])^2)))
}

## can be called for both single attribute value vectors and for the whole dataset
kernel.sigmoid <- function(av1, av2=av1, gamma=0.1, b=0)
{ tanh(gamma*(as.matrix(av1)%*%t(av2))+b) }

  # kernel functions called for instance pairs
kernel.linear(kmdat.train[1,1:4], kmdat.train[2,1:4])
kernel.polynomial(kmdat.train[1,1:4], kmdat.train[2,1:4])
kernel.radial(kmdat.train[1,1:4], kmdat.train[2,1:4])
kernel.sigmoid(kmdat.train[1,1:4], kmdat.train[2,1:4])

  # kernel functions called for the dataset (using the first 10 instances)
kernel.linear(kmdat.train[1:10,1:4])
kernel.polynomial(kmdat.train[1:10,1:4])
kernel.radial(kmdat.train[1:10,1:4])
kernel.sigmoid(kmdat.train[1:10,1:4])
##############
#16-6-1.R
##############
## predict using a kernel-based model
predict.kernel <- function(model, data)
{
  attributes <- x.vars(model$formula, data)
  aind <- names(data) %in% attributes
  amat <- as.matrix(data[,aind,drop=FALSE])
  kmat <- do.call(model$kernel, c(list(amat, model$mat), model$kernel.args))
  rowSums(cmm(kmat, model$coef))+model$intercept
}

  # kernel models for producing plots
kmplot <- list(coef=c(rep(1, 50), rep(-2, 50)),
                mat=as.matrix(kmdat.plot[sample(nrow(kmdat.plot), 100),1:2]),
                intercept=1, formula=f~a1+a2)
kmplot.l <- `class<-`(c(kmplot, kernel=kernel.linear), "kernel")
kmplot.p <- `class<-`(c(kmplot, kernel=kernel.polynomial), "kernel")
kmplot.r <- `class<-`(c(kmplot, kernel=kernel.radial), "kernel")
kmplot.s <- `class<-`(c(kmplot, kernel=kernel.sigmoid), "kernel")

  # generate predictions using different kernel functions
kmdat.plot$hl <- predict(kmplot.l, kmdat.plot)
kmdat.plot$hp <- predict(kmplot.p, kmdat.plot)
kmdat.plot$hr <- predict(kmplot.r, kmdat.plot)
kmdat.plot$hs <- predict(kmplot.s, kmdat.plot)

  # plot prediction surfaces
wf.kl <- wireframe(hl~a1+a2, kmdat.plot, col="grey50", zoom=0.8)
wf.kp <- wireframe(hp~a1+a2, kmdat.plot, col="grey50", zoom=0.8)
wf.kr <- wireframe(hr~a1+a2, kmdat.plot, col="grey50", zoom=0.8)
wf.ks <- wireframe(hs~a1+a2, kmdat.plot, col="grey50", zoom=0.8)

print(wf.kl, split=c(1, 1, 2, 2), more=TRUE)
print(wf.kp, split=c(2, 1, 2, 2), more=TRUE)
print(wf.kr, split=c(1, 2, 2, 2), more=TRUE)
print(wf.ks, split=c(2, 2, 2, 2))
##############
#16-7-1.R
##############
## kernel-based soft-margin SVM parameter estimation using quadratic programming
## solvers: "solve.QP" or "ipop"
svm.kernel <- function(formula, data, kernel=kernel.linear, kernel.args=NULL,
                       cost=1, svthres=1e-3, solver="solve.QP")
{
  class <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  cvec <- 2*as.num0(data[[class]])-1  # class vector using {-1, 1} labels
  ccmat <- outer(cvec, cvec)          # class-class product matrix
  amat <- as.matrix(data[,aind])      # attribute value matrix
  kmat <- do.call(kernel, c(list(amat), kernel.args)) # kernel matrix

  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(kmat*ccmat)$mat,
                 dvec=rep(1, nrow(data)),
                 Amat=matrix(c(cvec, diag(1, nrow(data)), diag(-1, nrow(data))),
                             nrow=nrow(data)),
                 bvec=c(0, rep(0, nrow(data)), rep(-cost, nrow(data))),
                 meq=1)
  else if (solver=="ipop")
    args <- list(c=rep(-1, nrow(data)),
                 H=kmat*ccmat,
                 A=cvec,
                 b=0,
                 l=rep(0, nrow(data)),
                 u=rep(cost, nrow(data)),
                 r=0)
  else
    stop("Unknown solver: ", solver)

  qp <- do.call(solver, args)
  alpha <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  sv <- which(alpha>svthres)
  model <- list(coef=cvec[sv]*alpha[sv], mat=amat[sv,,drop=FALSE],
                kernel=kernel, kernel.args=kernel.args, formula=formula)
  i <- which.min(abs(alpha-cost/2))
  'class<-'(c(model, intercept=cvec[i]-
                                 unname(predict.kernel(c(model, intercept=0),
                                                       data[i,aind,drop=FALSE]))),
            "svm.kernel")
}

## kernel-based SVM prediction
predict.svm.kernel <- function(model, data)
{
  ustep(predict.kernel(model, data))
}

  # kernel-based SVM for the artificial data
svm.kl <- svm.kernel(c~a1+a2+a3+a4, kmdat.train)
svm.kp <- svm.kernel(c~a1+a2+a3+a4, kmdat.train,
                     kernel=kernel.polynomial, kernel.args=list(p=2, b=1))
svm.kr <- svm.kernel(c~a1+a2+a3+a4, kmdat.train,
                     kernel=kernel.radial, kernel.args=list(gamma=0.5))
svm.ks <- svm.kernel(c~a1+a2+a3+a4, kmdat.train,
                     kernel=kernel.sigmoid, kernel.args=list(gamma=0.04,b=-0.8))

  # kernel-based SVM for the Pima Indians Diabetes
pid.svm.kl <- svm.kernel(diabetes~., pid.std.train)
pid.svm.kr <- svm.kernel(diabetes~., pid.std.train,
                         kernel=kernel.radial, kernel.args=list(gamma=0.1))

  # training set misclassification error
err(predict(svm.kl, kmdat.train), kmdat.train$c)
err(predict(svm.kp, kmdat.train), kmdat.train$c)
err(predict(svm.kr, kmdat.train), kmdat.train$c)
err(predict(svm.ks, kmdat.train), kmdat.train$c)

err(factor(predict(pid.svm.kl, pid.std.train[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.std.train$diabetes)
err(factor(predict(pid.svm.kr, pid.std.train[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.std.train$diabetes)

  # test set misclassification error
err(predict(svm.kl, kmdat.test), kmdat.test$c)
err(predict(svm.kp, kmdat.test), kmdat.test$c)
err(predict(svm.kr, kmdat.test), kmdat.test$c)
err(predict(svm.ks, kmdat.test), kmdat.test$c)

err(factor(predict(pid.svm.kl, pid.std.test[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.test$diabetes)
err(factor(predict(pid.svm.kr, pid.std.test[,-9]),
           levels=0:1, labels=levels(pid.std.train$diabetes)),
    pid.test$diabetes)
##############
#16-7-2.R
##############
## kernel-based SVR parameter estimation using quadratic programming
## solvers: "solve.QP" or "ipop"
svr.kernel <- function(formula, data, eps=0.01,
                       kernel=kernel.linear, kernel.args=NULL,
                       cost=1, svthres=1e-3, solver="solve.QP")
{
  f <- y.var(formula)
  attributes <- x.vars(formula, data)
  aind <- names(data) %in% attributes

  fvec <- data[[f]]                   # target function vector
  amat <- as.matrix(data[,aind])      # attribute value matrix
  kmat <- do.call(kernel, c(list(amat), kernel.args)) # kernel matrix

  if (solver=="solve.QP")
    args <- list(Dmat=nearPD(rbind(cbind(kmat, -kmat), cbind(-kmat, kmat)))$mat,
                 dvec=c(fvec-eps, -fvec-eps),
                 Amat=matrix(c(rep(1, nrow(data)), rep(-1, nrow(data)),
                               diag(1, 2*nrow(data)), diag(-1, 2*nrow(data))),
                             nrow=2*nrow(data)),
                 bvec=c(0, rep(0, 2*nrow(data)), rep(-cost, 2*nrow(data))),
                 meq=1)
  else if (solver=="ipop")
    args <- list(c=c(-fvec+eps, fvec+eps),
                 H=rbind(cbind(kmat, -kmat), cbind(-kmat, kmat)),
                 A=c(rep(1, nrow(data)), rep(-1, nrow(data))),
                 b=0,
                 l=rep(0, 2*nrow(data)),
                 u=rep(cost, 2*nrow(data)),
                 r=0)
  else
    stop("Unknown solver: ", solver)

  qp <- do.call(solver, args)
  alpha <- if (solver=="solve.QP") qp$solution else if (solver=="ipop") qp@primal
  beta <- alpha[1:nrow(data)]-alpha[(nrow(data)+1):(2*nrow(data))]
  sv <- which(abs(beta)>svthres)
  model <- list(coef=beta[sv], mat=amat[sv,,drop=FALSE],
                kernel=kernel, kernel.args=kernel.args, formula=formula)
  i <- which.min(abs(beta-cost/2))
  `class<-`(c(model,
              intercept=fvec[i]-unname(predict.kernel(c(model, intercept=0),
                                                      data[i,aind,drop=FALSE]))-
                          sign(beta[i])*eps),
            "svr.kernel")
}

## kernel-based SVR prediction
predict.svr.kernel <- predict.kernel

  # kernel-based SVR for f
svrf.kl <- svr.kernel(f~a1+a2+a3+a4, kmdat.train)
svrf.kp <- svr.kernel(f~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.polynomial, kernel.args=list(p=2, b=1))
svrf.kr <- svr.kernel(f~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.radial, kernel.args=list(gamma=0.02))
svrf.ks <- svr.kernel(f~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.sigmoid, kernel.args=list(gamma=0.2, b=0))

  # kernel-based SVR for g
svrg.kl <- svr.kernel(g~a1+a2+a3+a4, kmdat.train)
svrg.kp <- svr.kernel(g~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.polynomial, kernel.args=list(p=2, b=1))
svrg.kr <- svr.kernel(g~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.radial, kernel.args=list(gamma=0.1))
svrg.ks <- svr.kernel(g~a1+a2+a3+a4, kmdat.train,
                      kernel=kernel.sigmoid, kernel.args=list(gamma=0.02, b=-1))

  # kernel-based SVR for the Boston Housing data
bh.svr.kl <- svr.kernel(medv~., bh.std.train)
bh.svr.kr <- svr.kernel(medv~., bh.std.train,
                        kernel=kernel.radial, kernel.args=list(gamma=0.1))

  # training set MSE
mse(predict(svrf.kl, kmdat.train), kmdat.train$f)
mse(predict(svrf.kp, kmdat.train), kmdat.train$f)
mse(predict(svrf.kr, kmdat.train), kmdat.train$f)
mse(predict(svrf.ks, kmdat.train), kmdat.train$f)

mse(predict(svrg.kl, kmdat.train), kmdat.train$g)
mse(predict(svrg.kp, kmdat.train), kmdat.train$g)
mse(predict(svrg.kr, kmdat.train), kmdat.train$g)
mse(predict(svrg.ks, kmdat.train), kmdat.train$g)

mse(predict(bh.svr.kl, bh.std.train[,-13]), bh.std.train$medv)
mse(predict(bh.svr.kr, bh.std.train[,-13]), bh.std.train$medv)

  # test set MSE
mse(predict(svrf.kl, kmdat.test), kmdat.test$f)
mse(predict(svrf.kp, kmdat.test), kmdat.test$f)
mse(predict(svrf.kr, kmdat.test), kmdat.test$f)
mse(predict(svrf.ks, kmdat.test), kmdat.test$f)

mse(predict(svrg.kl, kmdat.test), kmdat.test$g)
mse(predict(svrg.kp, kmdat.test), kmdat.test$g)
mse(predict(svrg.kr, kmdat.test), kmdat.test$g)
mse(predict(svrg.ks, kmdat.test), kmdat.test$g)

mse(predict(bh.svr.kl, bh.std.test[,-13]), bh.std.test$medv)
mse(predict(bh.svr.kr, bh.std.test[,-13]), bh.std.test$medv)
##############
#17-1-1.R
##############
library(dmr.claseval)
library(dmr.util)

library(rpart)
library(e1071)

data(weatherc, package="dmr.data")
data(Vehicle, package="mlbench")
data(Glass, package="mlbench")

set.seed(12)

rv <- runif(nrow(Vehicle))
v.train <- Vehicle[rv>=0.33,]
v.test <- Vehicle[rv<0.33,]

rg <- runif(nrow(Glass))
g.train <- Glass[rg>=0.33,]
g.test <- Glass[rg<0.33,]

  # baseline models and their prediction
v.tree <- rpart(Class~., v.train)
v.tree.pred <- predict(v.tree, v.test, type="c")

g.tree <- rpart(Type~., g.train)
g.tree.pred <- predict(g.tree, g.test, type="c")

v.nb <- naiveBayes(Class~., v.train)
v.nb.pred <- predict(v.nb, v.test, type="c")

g.nb <- naiveBayes(Type~., g.train)
g.nb.pred <- predict(g.nb, g.test, type="c")
##############
#17-2-1.R
##############
## wrap single-attribute modeling transformation transm
## so that it is applied to all attributes for which condf returns TRUE
transmod.all <- function(transm, condf=function(v) TRUE)
{
  function(formula, data, ...)
  {
    attributes <- x.vars(formula, data)
    sapply(attributes,
           function(a) if (condf(data[[a]])) transm(data[[a]], ...),
           simplify=FALSE)
  }
}

## apply transformation model to a dataset
predict.transmod <- function(pred.transm)
{
  function(model, data, ...)
  {
    as.data.frame(sapply(names(data),
                         function(a)
                         if (a %in% names(model) && !is.null(model[[a]]))
                           pred.transm(model[[a]], data[[a]], ...)
                         else data[[a]],
                         simplify=FALSE))
  }
}

  # simple centering (mean subtraction) transformation
center.m <- transmod.all(mean, is.numeric)
  # performed on the weatherc data
w.cm <- center.m(play~., weatherc)
  # applied to the weatherc data
predict.center.m <- predict.transmod(function(m, v) v-m)
predict.center.m(w.cm, weatherc)
##############
#17-2-2.R
##############
transnonmod.all <- function(transnm, condf=function(v) TRUE)
{
  function(formula, data, ...)
  {
    attributes <- x.vars(formula, data)
    as.data.frame(sapply(names(data),
                         function(a)
                           if (a %in% attributes && condf(data[[a]]))
                             transnm(data[[a]], ...)
                           else data[[a]],
                         simplify=FALSE))
  }
}

  # simple centering (mean subtraction) transformation
center.nm <- transnonmod.all(function(v) v-mean(v), is.numeric)
  # performed on the weatherc data
center.nm(play~., weatherc)

  # simple round to a multiple of 10 transformation
divmod10 <- transnonmod.all(function(v) cbind(v %/% 10, v %% 10), is.numeric)
  # performed on the weatherc data
divmod10(play~., weatherc)
##############
#17-3-1.R
##############
## single-attribute standardization transformation
std <- function(v)
{
  list(mean=mean(v, na.rm=TRUE), sd=sd(v, na.rm=TRUE))
}

## standardization of all continuous attributes
std.all <- transmod.all(std, is.numeric)

## standardization model prediction
predict.std <- predict.transmod(function(m, v) (v-m$mean)/m$sd)

  # standardization model for the weatherc data
w.stdm <- std.all(play~., weatherc)
  # applied to the weatherc data
w.std <- predict.std(w.stdm, weatherc)

  # standardization model for the Glass data
g.stdm <- std.all(Type~., g.train)
  # applied to the training and test sets
g.train.std <- predict.std(g.stdm, g.train)
g.test.std <- predict.std(g.stdm, g.test)
##############
#17-3-2.R
##############
## single-attribute normalization transformation
nrm <- function(v)
{
  list(min=min(v, na.rm=TRUE), max=max(v, na.rm=TRUE))
}

## normalization of all continuous attributes
nrm.all <- transmod.all(nrm, is.numeric)

## standardization model prediction
predict.nrm <- predict.transmod(function(m, v) (v-m$min)/(m$max-m$min))

  # normalization model for the weatherc data
w.nrmm <- nrm.all(play~., weatherc)
  # applied to the weatherc data
w.nrm <- predict.nrm(w.nrmm, weatherc)

  # normalization model for the Glass data
g.nrmm <- nrm.all(Type~., g.train)
  # applied to the training and test sets
g.train.nrm <- predict.nrm(g.nrmm, g.train)
g.test.nrm <- predict.nrm(g.nrmm, g.test)
##############
#17-3-3.R
##############
## single-attribute aggregation transformation with m most frequent values retained
## and others replaced by comb.val
agg <- function(v, m, comb.val="other")
{
  list(retained=names(sort(table(v), decreasing=TRUE))[1:min(m, nlevels(v))],
       combined=comb.val)
}

## normalization of all discrete attributes
agg.all <- transmod.all(agg, is.factor)

## aggregation model prediction
predict.agg <- predict.transmod(function(m, v)
                                factor(ifelse(v %in% m$retained,
                                              as.character(v),
                                              ifelse(is.na(v), NA, m$combined)),
                                       levels=c(m$retained, m$combined)))

  # aggregation model for the weatherc data
w.aggm <- agg.all(play~., weatherc, 1)
  # applied to the weatherc data
w.agg <- predict.agg(w.aggm, weatherc)
##############
#17-3-4.R
##############
## single-attribute mean/median/mode imputation transformation
imp <- function(v, med=FALSE)
{
  if (!is.numeric(v))
    modal(v)
  else if (med)
    median(v, na.rm=TRUE)
  else
    mean(v, na.rm=TRUE)
}

## imputation for all attributes
imp.all <- transmod.all(imp)

## imputation model prediction
predict.imp <- predict.transmod(function(m, v) { v[is.na(v)] <- m; v } )

weathercm <- weatherc
weathercm$outlook[c(1, 3)] <- NA
weathercm$temperature[c(2, 4)] <- NA
weathercm$humidity[c(3, 5)] <- NA
weathercm$wind[c(4, 6)] <- NA

gm.train <- g.train
gm.train[sample.int(nrow(gm.train), 0.1*nrow(gm.train)),
         sample.int(ncol(gm.train)-1, 3)] <- NA
gm.test <- g.test
gm.test[sample.int(nrow(gm.test), 0.1*nrow(gm.test)),
        sample.int(ncol(gm.test)-1, 3)] <- NA

  # imputation model for the weatherc data
wm.impm <- imp.all(play~., weathercm, med=TRUE)
  # applied to the weatherc data
wm.imp <- predict.imp(wm.impm, weathercm)

  # imputation model for the Glass data
gm.impm <- imp.all(Type~., gm.train)
  # applied to the training and test sets
gm.train.imp <- predict.imp(gm.impm, gm.train)
gm.test.imp <- predict.imp(gm.impm, gm.test)
##############
#17-3-5.R
##############
## binary encoding of a single discrete attribute value
discode1 <- function(v, b=c(0,1), red=FALSE, na.all=FALSE)
{
  r <- 1-as.integer(red)
  if (is.factor(v) && ! (is.na(v) && na.all))
    b[1+as.integer(v==levels(v)[1:(nlevels(v)-r)])]
  else if (is.factor(v))
    rep(b[2], nlevels(v)-r)
  else
    v
}

## binary encoding of an discrete attribute
discode.a <- function(a, b=c(0,1), red=FALSE, na.all=FALSE)
{
  do.call(rbind, lapply(a, discode1, b, red, na.all))
}

## binary encoding of all discrete attributes in a given dataset
discode <- transnonmod.all(discode.a)

  # encoding a single attribute value
discode1(weatherc$outlook[1])
discode1(weatherc$outlook[1], red=TRUE)
discode1(factor(NA, levels=levels(weatherc$outlook)))
discode1(factor(NA, levels=levels(weatherc$outlook)), na.all=TRUE)

  # encoding single attributes of the weatherc data
discode.a(weatherc$outlook)
discode.a(weatherc$temperature)
discode.a(weatherc$outlook, b=c(-1,1), red=TRUE)
discode.a(weatherc$wind, b=c(-1,1))

  # encoding single instances of the weatherc data
discode(~., weatherc[1,])
discode(~., weatherc[1,], red=TRUE)

  # encoding the complete weatherc data
discode(~., weatherc)
  # leave the target attribute unchanged
discode(play~., weatherc)
##############
#17-4-1.R
##############
## natural binary multiclass encoding function
multi.enc.nbc <- function(d, class, clabs=levels(d))
{
  `colnames<-`(ed <- t(sapply(as.integer(d), int2binvec, length(clabs))),
               paste(class, 1:ncol(ed), sep="."))
}

## natural binary multiclass decoding function
multi.dec.nbc <- function(pred, clabs)
{
  clabs[clip.val(apply(pred, 1, binvec2int), 1, length(clabs))]
}

## generate a multiclass wrapper around alg using predf for prediction
## and the specified encoding and decoding functions
multi.class <- function(alg, predf=predict,
                        encode=multi.enc.nbc, decode=multi.dec.nbc)
{
  mc.alg <- function(formula, data, ...)
  {
    attributes <- x.vars(formula, data)
    class <- y.var(formula)
    aind <- names(data) %in% attributes
    clabs <- flevels(data[[class]])

    d.enc <- encode(data[[class]], class, clabs)
    binmodels <- lapply(1:ncol(d.enc),
                        function(i)
                        alg(make.formula(colnames(d.enc)[i], attributes),
                            `names<-`(cbind(data[,aind],
                                            factor(d.enc[,i], levels=0:1)),
                                      c(names(data[,aind]), colnames(d.enc)[i])),
                            ...))
    list(binmodels=binmodels, clabs=clabs)
  }

  mc.predict <- function(model, data, ...)
  {
    decode(sapply(model$binmodels, predf, data, ...), model$clabs)
  }

  list(alg=mc.alg, predict=mc.predict)
}

  # encoding class labels and decoding verification
v.nbc <- multi.enc.nbc(Vehicle$Class, "Class")
err(multi.dec.nbc(v.nbc, levels(Vehicle$Class)), Vehicle$Class)

  # basic encoding applied to rpart
rp.n <- multi.class(rpart, predf=function(...) predict(..., type="c"))
v.tree.n <- rp.n$alg(Class~., v.train)
v.tree.n.pred <- rp.n$predict(v.tree.n, v.test)
g.tree.n <- rp.n$alg(Type~., g.train)
g.tree.n.pred <- rp.n$predict(g.tree.n, g.test)

  # basic encoding applied to naive Bayes
nb.n <- multi.class(naiveBayes)
v.nb.n <- nb.n$alg(Class~., v.train)
v.nb.n.pred <- nb.n$predict(v.nb.n, v.test)
g.nb.n <- nb.n$alg(Type~., g.train)
g.nb.n.pred <- nb.n$predict(g.nb.n, g.test)
##############
#17-4-2.R
##############
## 1-of-k multiclass encoding function
multi.enc.1ofk <- function(d, class, clabs=levels(d))
{
  `colnames<-`(sapply(clabs, function(cl) as.integer(d==cl)),
               paste(class, clabs, sep="."))
}

## 1-of-k multiclass encoding function
multi.dec.1ofk <- function(pred, clabs)
{
  clabs[max.col(pred)]
}

  # encoding class labels and decoding verification
v.1ofk <- multi.enc.1ofk(Vehicle$Class, "Class")
err(multi.dec.1ofk(v.1ofk, levels(Vehicle$Class)), Vehicle$Class)

  # 1-of-k encoding applied to rpart
rp.1 <- multi.class(rpart, predf=function(...) predict(...)[,2],
                    encode=multi.enc.1ofk, decode=multi.dec.1ofk)
v.tree.1 <- rp.1$alg(Class~., v.train)
v.tree.1.pred <- rp.1$predict(v.tree.1, v.test)
g.tree.1 <- rp.1$alg(Type~., g.train)
g.tree.1.pred <- rp.1$predict(g.tree.1, g.test)

  # 1-of-k encoding applied to naive Bayes
nb.1 <- multi.class(naiveBayes, predf=function(...) predict(..., type="r")[,2],
                    encode=multi.enc.1ofk, decode=multi.dec.1ofk)
v.nb.1 <- nb.1$alg(Class~., v.train)
v.nb.1.pred <- nb.1$predict(v.nb.1, v.test)
g.nb.1 <- nb.1$alg(Type~., g.train)
g.nb.1.pred <- nb.1$predict(g.nb.1, g.test)
##############
#17-4-3.R
##############
## error correcting codeword for i out of k
ecc <- function(i, k)
{
  if (i>1)
    c(rev(1-c(ecc(i-1, k-1), 1)), ecc(i-1,k-1))
  else
    rep(1, 2^(k-1)-1)
}

  # error-correcting code for 3 codewords
t(sapply(1:3, ecc, 3))
  # error-correcting code for 4 codewords
t(sapply(1:4, ecc, 4))
  # error-correcting code for 5 codewords
t(sapply(1:5, ecc, 5))
##############
#17-4-4.R
##############
## error-correcting multiclass encoding and decoding function generator
## for k classes with up to maxbits bits
multi.ecc <- function(k, maxbits=Inf)
{
  code <- sapply(1:k, ecc, k)
  if (nrow(code)>maxbits)
    code <- code[sample.int(nrow(code), maxbits),]

  enc <- function(d, class, clabs=levels(d))
  {
    `colnames<-`(t(code[,d]), paste(class, 1:nrow(code), sep = "."))
  }

  dec <- function(pred, clabs)
  {
    clabs[apply(pred, 1, function(p) which.min(colSums(p!=code)))]
  }
  list(enc=enc, dec=dec)
}

  # error correcting encoding/decoding for 4 and 6 classes
multi.ecc4 <- multi.ecc(4)
multi.ecc6 <- multi.ecc(6)

  # error-correcding encoding applied to rpart
rp.e4 <- multi.class(rpart, predf=function(...) predict(..., type="c"),
                     encode=multi.ecc4$enc, decode=multi.ecc4$dec)
rp.e6 <- multi.class(rpart, predf=function(...) predict(..., type="c"),
                     encode=multi.ecc6$enc, decode=multi.ecc6$dec)
v.tree.e <- rp.e4$alg(Class~., v.train)
v.tree.e.pred <- rp.e4$predict(v.tree.e, v.test)
g.tree.e <- rp.e6$alg(Type~., g.train)
g.tree.e.pred <- rp.e6$predict(g.tree.e, g.test)

  # error-correcting encoding applied to naive Bayes
nb.e4 <- multi.class(naiveBayes, encode=multi.ecc4$enc, decode=multi.ecc4$dec)
nb.e6 <- multi.class(naiveBayes, encode=multi.ecc6$enc, decode=multi.ecc6$dec)

v.nb.e <- nb.e4$alg(Class~., v.train)
v.nb.e.pred <- nb.e4$predict(v.nb.e, v.test)
g.nb.e <- nb.e6$alg(Type~., g.train)
g.nb.e.pred <- nb.e6$predict(g.nb.e, g.test)
##############
#17-4-5.R
##############
v.tree.err <- c(direct=err(v.tree.pred, v.test$Class),
                nbc=err(v.tree.n.pred, v.test$Class),
                `1ofk`=err(v.tree.1.pred, v.test$Class),
                ecc=err(v.tree.e.pred, v.test$Class))

v.nb.err <- c(direct=err(v.nb.pred, v.test$Class),
                nbc=err(v.nb.n.pred, v.test$Class),
                `1ofk`=err(v.nb.1.pred, v.test$Class),
                ecc=err(v.nb.e.pred, v.test$Class))

g.tree.err <- c(direct=err(g.tree.pred, g.test$Type),
                nbc=err(g.tree.n.pred, g.test$Type),
                `1ofk`=err(g.tree.1.pred, g.test$Type),
                ecc=err(g.tree.e.pred, g.test$Type))

g.nb.err <- c(direct=err(g.nb.pred, g.test$Type),
              nbc=err(g.nb.n.pred, g.test$Type),
              `1ofk`=err(g.nb.1.pred, g.test$Type),
              ecc=err(g.nb.e.pred, g.test$Type))

barplot(v.tree.err, main="Vehicle Silhouettes, rpart", ylab="Error", las=2)
lines(c(0, 5), rep(v.tree.err[1], 2), lty=2)

barplot(v.nb.err, main="Vehicle Silhouettes, naiveBayes", ylab="Error", las=2)
lines(c(0, 5), rep(v.nb.err[1], 2), lty=2)

barplot(g.tree.err, main="Glass, rpart", ylab="Error", las=2)
lines(c(0, 5), rep(g.tree.err[1], 2), lty=2)

barplot(g.nb.err, main="Glass, naiveBayes", ylab="Error", las=2)
lines(c(0, 5), rep(g.nb.err[1], 2), lty=2)
##############
#18-1-1.R
##############
library(dmr.claseval)
library(dmr.stats)
library(dmr.trans)
library(dmr.util)

library(rpart)
library(e1071)

data(weatherc, package="dmr.data")
data(Vehicle, package="mlbench")
data(Glass, package="mlbench")

set.seed(12)

rv <- runif(nrow(Vehicle))
v.train <- Vehicle[rv>=0.33,]
v.test <- Vehicle[rv<0.33,]

rg <- runif(nrow(Glass))
g.train <- Glass[rg>=0.33,]
g.test <- Glass[rg<0.33,]
##############
#18-2-1.R
##############
## create a wrapper for all attributes discretization
disc.all <- function(disc)
{
  disc1 <- function(v, k, class, ...)
  {
    if (is.numeric(v))
    {
      if (is.null(formals(disc)$class))
        disc(v, k=k, ...)  # unsupervised
      else
        disc(v, k=k, class=class, ...)  # supervised
    }
  }

  function(formula, data, k=5, ...)
  {
    attributes <- x.vars(formula, data)
    class <- y.var(formula)
    if (length(k)==1)
      k <- sapply(attributes, function(a) k)
    km <- match(attributes, names(k))
    `class<-`(mapply(function(a, k1) disc1(data[[a]], k=k1, class=data[[class]], ...),
                     attributes, k[km], SIMPLIFY=FALSE),
              "disc")
  }
}

## apply discretization breaks to a dataset
predict.disc <- predict.transmod(function(m, v) cut(v, c(-Inf, m, Inf)))

  # random all-attributes discretization
disc.rand <- disc.all(function(v, k=3) sort(runif(k-1, min=min(v), max=max(v))))

  # random discretization for the weatherc data
w.dr3m <- disc.rand(play~., weatherc, 3)
w.dr43m <- disc.rand(play~., weatherc, list(temperature=4, humidity=3))

  # apply discretization breaks to the weatherc data
w.dr3 <- predict(w.dr3m, weatherc)
w.dr43 <- predict(w.dr43m, weatherc)
##############
#18-3-1.R
##############
## equal-width discretization for a single attribute
disc.eqwidth1 <- function(v, k=5)
{
  w <- diff(r <- range(v))/k
  seq(r[1]+w, r[2]-w, w)
}

## equal-width discretization for a dataset
disc.eqwidth <- disc.all(disc.eqwidth1)

  # equal-width discretization of the temperature attribute in the weatherc data
disc.eqwidth1(weatherc$temperature, 4)

  # equal-width discretization for the weatherc data
disc.eqwidth(play~., weatherc, 3)
disc.eqwidth(play~., weatherc, list(temperature=4, humidity=3))

  # equal-width discretization for the Vehicle Silhouettes data
v.disc.ew <- disc.eqwidth(Class~., v.train, 7)
summary(predict(v.disc.ew, v.train))

  # equal-width discretization for the Glass data
g.disc.ew <- disc.eqwidth(Type~., g.train, 7)
summary(predict(g.disc.ew, g.train))
##############
#18-3-2.R
##############
## equal-frequency discretization for a single attribute
disc.eqfreq1 <- function(v, k=5) { unique(quantile(v, seq(1/k, 1-1/k, 1/k))) }

## equal-frequency discretization for a dataset
disc.eqfreq <- disc.all(disc.eqfreq1)

  # equal-width discretization of the temperature attribute in the weatherc data
disc.eqfreq1(weatherc$temperature, 4)

  # equal-frequency discretization for the weatherc data
disc.eqfreq(play~., weatherc, 3)
disc.eqfreq(play~., weatherc, list(temperature=4, humidity=3))

  # equal-frequency discretization for the Vehicle Silhouettes data
v.disc.ef <- disc.eqfreq(Class~., v.train, 7)
summary(predict(v.disc.ef, v.train))

  # equal-frequency discretization for the Glass data
g.disc.ef <- disc.eqfreq(Type~., g.train, 7)
summary(predict(g.disc.ef, g.train))
##############
#18-3-3.R
##############
## create a non-modeling discretization wrapper
disc.nm <- function(disc)
{ function(formula, data, k=5, ...) predict(disc(formula, data, k, ...), data) }

## non-modeling equal-width discretization
discnm.eqwidth <- disc.nm(disc.eqwidth)

## non-modeling equal-frequency discretization
discnm.eqfreq <- disc.nm(disc.eqfreq)

  # non-modeling discretization for the weatherc data
discnm.eqwidth(play~., weatherc, 4)
discnm.eqfreq(play~., weatherc, 3)
discnm.eqfreq(play~., weatherc, list(temperature=4, humidity=3))
##############
#18-4-1.R
##############
## pure-class discretization for a single attribute
disc.pure1 <- function(v, class, k=NULL)
{
  ord <- order(v)
  class <- class[ord]
  v <- v[ord]
  b <- diff(as.integer(class))!=0 & diff(v)!=0
  (v[1:(length(v)-1)][b]+v[2:length(v)][b])/2
}

## pure-class discretization for a dataset
disc.pure <- disc.all(disc.pure1)

  # pure-class discretization for the weatherc data
disc.pure(play~., weatherc)

  # pure-class discretization for the Vehicle Silhouettes data
v.disc.p <- disc.pure(Class~., v.train)
summary(predict(v.disc.p, v.train), maxsum=100)

  # pure-class discretization for the Glass data
g.disc.p <- disc.pure(Type~., g.train)
summary(predict(g.disc.p, g.train), maxsum=100)
##############
#18-4-2.R
##############
## bottom-up discretization for a single attribute
disc.bottomup1 <- function(v, class, k=5, initf=disc.pure1, evalf, maxev=Inf)
{
  breaks <- initf(v, class)
  utils <- mapply(evalf, breaks,
                  shift.right(breaks, first=-Inf), shift.left(breaks, last=Inf),
                  MoreArgs=list(v, class))
  b <- which.min(utils)

  while (length(breaks)+1>k && utils[b]<=maxev)
  {
    breaks <- breaks[-b]
    utils <- utils[-b]
    if (b>1)
      utils[b-1] <- evalf(breaks[b-1],
                          ifelse(b>2, breaks[b-2], -Inf),
                          ifelse(b<=length(breaks), breaks[b], Inf),
                          v, class)
    if (b<=length(breaks))
      utils[b] <- evalf(breaks[b],
                        ifelse(b>1, breaks[b-1], -Inf),
                        ifelse(b<length(breaks), breaks[b+1], Inf),
                        v, class)

    b <- which.min(utils)
  }
  breaks
}

## bottom-up discretization for a dataset
disc.bottomup <- disc.all(disc.bottomup1)

  # bottom-up discretization of the temperature attribute in the weatherc data
disc.bottomup1(weatherc$temperature, weatherc$play, 3,
               evalf=function(b, bl, br, v, class) b)

  # bottom-up discretization for the weatherc data
disc.bottomup(play~., weatherc, 3, evalf=function(b, bl, br, v, class) b)
##############
#18-4-3.R
##############
## instance count break evaluation
## using the total instance count after merging
evdisc.incount1 <- function(b, bl, br, v, class) { sum(v>bl & v<=br) }

## instance count break evaluation
## using the minimum individual instance count before merging
evdisc.incount2 <- function(b, bl, br, v, class, gamma=1/length(v))
{
  min(sum(v>bl & v<=b), sum(v>b & v<=br)) +
    gamma*max(sum(v>bl & v<=b), sum(v>b & v<=br))
}

  # instance count bottom-up discretization for the weatherc data
disc.bottomup(play~., weatherc, 3, evalf=evdisc.incount1)
disc.bottomup(play~., weatherc, 3, evalf=evdisc.incount2)

  # instance count bottom-up discretization for the Vehicle Silhouettes data
v.disc.bu.ic1 <- disc.bottomup(Class~., v.train, 7, evalf=evdisc.incount1)
v.disc.bu.ic2 <- disc.bottomup(Class~., v.train, 7, evalf=evdisc.incount2)
summary(predict(v.disc.bu.ic1, v.train))
summary(predict(v.disc.bu.ic2, v.train))

  # instance count bottom-up discretization for the Glass data
g.disc.bu.ic1 <- disc.bottomup(Type~., g.train, 7, evalf=evdisc.incount1)
g.disc.bu.ic2 <- disc.bottomup(Type~., g.train, 7, evalf=evdisc.incount2)
summary(predict(g.disc.bu.ic1, g.train))
summary(predict(g.disc.bu.ic2, g.train))
##############
#18-4-4.R
##############
## dominating class count break evaluation
## using the total dominating class count after merging
evdisc.dccount1 <- function(b, bl, br, v, class) { max(table(class[v>bl & v<=br])) }

## dominating class count break evaluation
## using the sum of individual dominating class counts before merging
evdisc.dccount2 <- function(b, bl, br, v, class)
{
  max(table(class[v>bl & v<=b])) + max(table(class[v>b & v<=br]))
}

## dominating class count break evaluation
## using the minimum individual dominating class count before merging
evdisc.dccount3 <- function(b, bl, br, v, class, gamma=1/length(v))
{
  ccl <- table(class[v>bl & v<=b])  # class counts: left
  ccr <- table(class[v>b & v<=br])  # and right

  dcl <- which.max(ccl)  # dominating classes: left
  dcr <- which.max(ccr)  # and right

  dcmin <- ifelse(ccl[dcl]<=ccr[dcr], dcl, dcr)
  min(ccl[dcl], ccr[dcr]) - gamma*max(ccl[dcmin], ccr[dcmin])
}

  # dominating class count bottom-up discretization for the weatherc data
disc.bottomup(play~., weatherc, 3, evalf=evdisc.dccount1)
disc.bottomup(play~., weatherc, 3, evalf=evdisc.dccount2)
disc.bottomup(play~., weatherc, 3, evalf=evdisc.dccount3)

  # dominating class count bottom-up discretization for the Vehicle Silhouettes data
v.disc.bu.dc1 <- disc.bottomup(Class~., v.train, 7, evalf=evdisc.dccount1)
v.disc.bu.dc2 <- disc.bottomup(Class~., v.train, 7, evalf=evdisc.dccount2)
v.disc.bu.dc3 <- disc.bottomup(Class~., v.train, 7, evalf=evdisc.dccount3)
summary(predict(v.disc.bu.dc1, v.train))
summary(predict(v.disc.bu.dc2, v.train))
summary(predict(v.disc.bu.dc3, v.train))

  # dominating class count bottom-up discretization for the Glass data
g.disc.bu.dc1 <- disc.bottomup(Type~., g.train, 7, evalf=evdisc.dccount1)
g.disc.bu.dc2 <- disc.bottomup(Type~., g.train, 7, evalf=evdisc.dccount2)
g.disc.bu.dc3 <- disc.bottomup(Type~., g.train, 7, evalf=evdisc.dccount3)
summary(predict(v.disc.bu.dc1, g.train))
summary(predict(v.disc.bu.dc2, g.train))
summary(predict(v.disc.bu.dc3, g.train))
##############
#18-4-5.R
##############
## misclassification count break evaluation
evdisc.mcount <- function(b, bl, br, v, class)
{
  mcount <- function(cond) { sum(cond) - max(table(class[cond])) }

  mcount(v>bl & v<=br) - (mcount(v>bl & v<=b) + mcount(v>b & v<=br))
}

  # misclassification count bottom-up discretization for the weatherc data
disc.bottomup(play~., weatherc, 3, evalf=evdisc.mcount)

  # misclassification count bottom-up discretization for the Vehicle Silhouettes data
v.disc.bu.mc <- disc.bottomup(Class~., v.train, 7, evalf=evdisc.mcount)
summary(predict(v.disc.bu.mc, v.train))

  # misclassification count bottom-up discretization for the Glass data
g.disc.bu.mc <- disc.bottomup(Type~., g.train, 7, evalf=evdisc.mcount)
summary(predict(g.disc.bu.mc, g.train))
##############
#18-4-6.R
##############
## entropy break evaluation
evdisc.entropy <- function(b, bl, br, v, class)
{
  sum(v>bl & v<=br)*(entropy(class[v>bl & v<=br])-
                       entropy.cond(class[v>bl & v<=br], v[v>bl & v<=br]<=b))
}

  # entropy bottom-up discretization for the weatherc data
disc.bottomup(play~., weatherc, 3, evalf=evdisc.entropy)

  # entropy bottom-up discretization for the Vehicle Silhouettes data
v.disc.bu.e <- disc.bottomup(Class~., v.train, 7, evalf=evdisc.entropy)
summary(predict(v.disc.bu.e, v.train))

  # entropy bottom-up discretization for the Glass data
g.disc.bu.e <- disc.bottomup(Type~., g.train, 7, evalf=evdisc.entropy)
summary(predict(g.disc.bu.e, g.train))
##############
#18-4-7.R
##############
## chi-square break evaluation
evdisc.chisq <- function(b, bl, br, v, class)
{
  chisq.test(class[v>bl & v<=br], v[v>bl & v<=br]<=b, correct=FALSE)$statistic
}

  # chi-square bottom-up discretization for the weatherc data
disc.bottomup(play~., weatherc, 3,evalf=evdisc.chisq)

  # chi-square bottom-up discretization for the Vehicle Silhouettes data
v.disc.bu.chi <- disc.bottomup(Class~., v.train, 7, evalf=evdisc.chisq)
summary(predict(v.disc.bu.chi, v.train))

  # chi-square bottom-up discretization for the Glass data
g.disc.bu.chi <- disc.bottomup(Type~., g.train, 7, evalf=evdisc.chisq)
summary(predict(g.disc.bu.chi, g.train))
##############
#18-4-8.R
##############
## top-down discretization for a single attribute
disc.topdown1 <- function(v, class, k=5, candf=disc.pure1, evalf, minev=0)
{
  evalf.td <- function(b, bl, br, v, class)
  {
    if (any(v>bl & v<=br))
     evalf(b, bl, br, v, class)
    else
      -Inf
  }

  breaks <- NULL
  candidates <- candf(v, class)

  utils <- mapply(evalf.td, candidates,
                  closest.below(candidates, breaks),
                  closest.above(candidates, breaks),
                  MoreArgs=list(v, class))
  b <- which.max(utils)

  while (length(candidates)>0 && length(breaks)+1<k && utils[b]>=minev)
  {
    breaks <- insert.ord(breaks, candidates[b])
    candidates <- candidates[-b]
    utils <- utils[-b]
    if (b>1)
      utils[b-1] <- evalf.td(candidates[b-1],
                             closest.below(candidates[b-1], breaks),
                             closest.above(candidates[b-1], breaks),
                             v, class)
    if (b<=length(candidates))
      utils[b] <- evalf.td(candidates[b],
                           closest.below(candidates[b], breaks),
                           closest.above(candidates[b], breaks),
                           v, class)
    b <- which.max(utils)
  }
  breaks
}

## top-down discretization for a dataset
disc.topdown <- disc.all(disc.topdown1)

  # top-down discretization of the temperature attribute in the weatherc data
disc.topdown1(weatherc$temperature, weatherc$play, 3,
              evalf=function(b, bl, br, v, class) b)

  # top-down discretization for the weatherc data
disc.topdown(play~., weatherc, 3, evalf=function(b, bl, br, v, class) b)
##############
#18-4-9.R
##############
   # misclassification count top-down discretization for the weatherc data
disc.topdown(play~., weatherc, 3, evalf=evdisc.mcount)
   # misclassification count top-down discretization for the Vehicle Silhouettes data
v.disc.td.mc <- disc.topdown(Class~., v.train, 7, evalf=evdisc.mcount)
summary(predict.disc(v.disc.td.mc, v.train))
   # misclassification count top-down discretization for the Glass data
g.disc.td.mc <- disc.topdown(Type~., g.train, 7, evalf=evdisc.mcount)
summary(predict.disc(g.disc.td.mc, g.train))

    # entropy top-down discretization for the weatherc data
disc.topdown(play~., weatherc, 3, evalf=evdisc.entropy)
   # entropy top-down discretization for the Vehicle Silhouettes data
v.disc.td.e <- disc.topdown(Class~., v.train, 7, evalf=evdisc.entropy)
summary(predict.disc(v.disc.td.e, v.train))
   # entropy top-down discretization for the Glass data
g.disc.td.e <- disc.topdown(Type~., g.train, 7, evalf=evdisc.entropy)
summary(predict.disc(g.disc.td.e, g.train))

    # chi-square top-down discretization for the weatherc data
disc.topdown(play~., weatherc, 3, evalf=evdisc.chisq)
   # chi-square top-down discretization for the Vehicle Silhouettes data
v.disc.td.chi <- disc.topdown(Class~., v.train, 7, evalf=evdisc.chisq)
summary(predict.disc(v.disc.td.chi, v.train))
   # chi-square top-down discretization for the Glass data
g.disc.td.chi <- disc.topdown(Type~., g.train, 7, evalf=evdisc.chisq)
summary(predict.disc(g.disc.td.chi, g.train))
##############
#18-5-1.R
##############
  # discretization models for the Vehicle Silhouettes data
v.disc <- list(nodisc=`class<-`(list(), "disc"),
               ew=v.disc.ew, ef=v.disc.ef,
               bu.ic1=v.disc.bu.ic1, bu.ic2=v.disc.bu.ic2,
               bu.dc1=v.disc.bu.dc1, bu.dc2=v.disc.bu.dc2, bu.dc3=v.disc.bu.dc3,
               bu.mc=v.disc.bu.mc, bu.e=v.disc.bu.e, bu.chi=v.disc.bu.chi,
               td.mc=v.disc.td.mc, td.e=v.disc.td.e, td.chi=v.disc.td.chi)

  # discretization models for the Glass data
g.disc <- list(nodisc=`class<-`(list(), "disc"),
               ew=g.disc.ew, ef=g.disc.ef,
               bu.ic1=g.disc.bu.ic1, bu.ic2=g.disc.bu.ic2,
               bu.dc1=g.disc.bu.dc1, bu.dc2=g.disc.bu.dc2, bu.dc3=g.disc.bu.dc3,
               bu.mc=g.disc.bu.mc, bu.e=g.disc.bu.e, bu.chi=g.disc.bu.chi,
               td.mc=g.disc.td.mc, td.e=g.disc.td.e, td.chi=g.disc.td.chi)

  # misclassification error values for the Vehicle Silhouettes data
v.err <- lapply(v.disc,
                function(dm)
                {
                  v.train.d <- predict(dm, v.train)
                  v.test.d <- predict(dm, v.test)
                  v.tree.d <- rpart(Class~., v.train.d)
                  v.nb.d <- naiveBayes(Class~., v.train.d)
                  list(tree=err(predict(v.tree.d, v.test.d, type="c"),
                                v.test.d$Class),
                       nb=err(predict(v.nb.d, v.test.d), v.test.d$Class))
                })

  # misclassification error values for the Glass data
g.err <- lapply(g.disc,
                function(dm)
                {
                  g.train.d <- predict(dm, g.train)
                  g.test.d <- predict(dm, g.test)
                  g.tree.d <- rpart(Type~., g.train.d)
                  g.nb.d <- naiveBayes(Type~., g.train.d)
                  list(tree=err(predict(g.tree.d, g.test.d, type="c"),
                                g.test.d$Type),
                       nb=err(predict(g.nb.d, g.test.d), g.test.d$Type))
                })

  # error comparison
v.tree.err <- sapply(v.err, function(e) e$tree)
g.tree.err <- sapply(g.err, function(e) e$tree)
v.nb.err <- sapply(v.err, function(e) e$nb)
g.nb.err <- sapply(g.err, function(e) e$nb)

barplot(v.tree.err, main="Vehicle Silhouettes, rpart", ylab="Error", las=2)
lines(c(0, 17), rep(v.tree.err[1], 2), lty=2)

barplot(g.tree.err, main="Glass, rpart", ylab="Error", las=2)
lines(c(0, 17), rep(g.tree.err[1], 2), lty=2)

barplot(v.nb.err, main="Vehicle Silhouettes, naiveBayes", ylab="Error", las=2)
lines(c(0, 17), rep(v.nb.err[1], 2), lty=2)

barplot(g.nb.err, main="Glass, naiveBayes", ylab="Error", las=2)
lines(c(0, 17), rep(g.nb.err[1], 2), lty=2)
##############
#19-1-1.R
##############
library(dmr.claseval)
library(dmr.disc)
library(dmr.regeval)
library(dmr.stats)
library(dmr.trans)
library(dmr.util)

library(rpart)

data(weather, package="dmr.data")
data(weatherc, package="dmr.data")
data(weatherr, package="dmr.data")

data(Vehicle, package="mlbench")
data(Soybean, package="mlbench")
data(BostonHousing, package="mlbench")

set.seed(12)

rv <- runif(nrow(Vehicle))
v.train <- Vehicle[rv>=0.33,]
v.test <- Vehicle[rv<0.33,]

rs <- runif(nrow(Soybean))
s.train <- Soybean[rs>=0.33,]
s.test <- Soybean[rs<0.33,]

rbh <- runif(nrow(BostonHousing))
bh.train <- BostonHousing[rbh>=0.33,]
bh.test <- BostonHousing[rbh<0.33,]
##############
#19-3-1.R
##############
## attribute selection search initialization
asel.init.none <- function(attributes) { character(0) }
asel.init.all <- function(attributes) { attributes }

  # attribute selection search initialization for the weather data
asel.init.none(names(weather)[-5])
asel.init.all(names(weather)[-5])
##############
#19-3-2.R
##############
## forward attribute selection search next state generation
asel.next.forward <- function(subset, attributes)
{
  lapply(setdiff(attributes, subset), function(a) c(subset, a))
}

## backward attribute selection search next state generation
asel.next.backward <- function(subset, attributes)
{
  if (length(subset)>1)
    lapply(1:length(subset), function(i) subset[-i])
  else
    list()
}

  # attribute selection next state generation for the weather data
asel.next.forward(c("outlook", "humidity"), names(weather)[-5])
asel.next.backward(c("outlook", "humidity"), names(weather)[-5])
##############
#19-3-3.R
##############
## greedy attribute selection search
asel.search.greedy <- function(attributes, target, evalf,
                               initf=asel.init.all, nextf=asel.next.backward,
                               max.noimp=3, penalty=0.01)
{
  ev <- function(subset)
  {
    ifelse(is.finite(v<-evalf(subset, target)), v-penalty*length(subset)*abs(v), v)
  }

  best.subset <- subset <- initf(attributes)
  best.eval <- eval <- ev(subset)
  noimp <- 0

  while (noimp < max.noimp)
  {
    candidates <- nextf(subset, attributes)
    if (length(candidates)>0)
    {
      cand.eval <- sapply(candidates, ev)
      cand.best <- which.max(cand.eval)
      noimp <- ifelse(cand.eval[cand.best]>eval, 0, noimp+1)
      subset <- candidates[[cand.best]]
      eval <- cand.eval[cand.best]
      if (eval>best.eval)
      {
        best.subset <- subset
        best.eval <- eval
      }
    }
    else
      break
  }

  list(subset=best.subset, eval=best.eval)
}

  # greedy attribute selection search for the weather data
asel.search.greedy(names(weather)[-5], "play", evalf=function(subset, target) 1)
asel.search.greedy(names(weather)[-5], "play", evalf=function(subset, target) 1,
                   penalty=0)
asel.search.greedy(names(weather)[-5], "play", evalf=function(subset, target) 1,
                   initf=asel.init.none, nextf=asel.next.forward)
asel.search.greedy(names(weather)[-5], "play", evalf=function(subset, target) 1,
                   initf=asel.init.none, nextf=asel.next.forward, penalty=0)
##############
#19-4-1.R
##############
dd.chi2 <- function(a1, a2) 1-chisq.test(a1, a2)$p.value
cd.kruskal <- function(a1, a2) 1-kruskal.test(a1, a2)$p.value
cc.spearman <- function(a1, a2) 1-cor.test(a1, a2, method="spearman")$p.value

## simple statistical attribute selection filter
simple.filter <- function(formula, data, dd=dd.chi2, cd=cd.kruskal, cc=cc.spearman)
{
  attributes <- x.vars(formula, data)
  target <- y.var(formula)

  utility <- function(a)
  {
    unname(switch(attr.type(data[[a]], data[[target]]),
                  dd = dd(data[[a]], data[[target]]),
                  cd = cd(data[[a]], data[[target]]),
                  dc = cd(data[[target]], data[[a]]),
                  cc = cc(data[[a]], data[[target]])))
  }

  sort(sapply(attributes, utility), decreasing=TRUE)
}

  # simple filter for the weather data
simple.filter(play~., weather)
simple.filter(play~., weather, dd=symunc)

  # simple filter for the weatherc data
simple.filter(play~., weatherc)
simple.filter(play~outlook+wind, weatherc, dd=symunc)
simple.filter(play~temperature+humidity, weatherc,
              cd=function(a1, a2) kruskal.test(a1, a2)$statistic)

  # simple filter for the weatherr data
simple.filter(playability~., weatherr)
simple.filter(playability~outlook+wind, weatherr,
              cd=function(a1, a2) kruskal.test(a1, a2)$statistic)
simple.filter(playability~temperature+humidity, weatherr,
              cc=function(a1, a2) cor(a1, a2, method="spearman")^2)

  # simple filter for the Vehicle Silhouettes data
v.utl.simple <- simple.filter(Class~., discnm.eqfreq(~., v.train, 7), dd=symunc)
  # simple filter for the Soybean data
s.utl.simple <- simple.filter(Class~., Soybean, dd=symunc)
  # simple filter for the BostonHousing data
bh.utl.simple <- simple.filter(medv~., discnm.eqfreq(~., bh.train, 7), dd=symunc)
##############
#19-4-2.R
##############
## discrete attribute correlation using binary encoding
discor <- function(a1, a2,
                   corf=function(a1, a2)
                        abs(cor(a1, a2, method="spearman", use="complete.obs")))
{
  switch(attr.type(a1, a2),
         cc=p12<-1,
         dc=p12<-as.matrix(pdisc(a1)),
         cd=p12<-t(as.matrix(pdisc(a2))),
         dd=p12<-pdisc(a1, a2))

  a1dc <- discode.a(a1, red=TRUE, na.all=TRUE)
  a2dc <- discode.a(a2, red=TRUE, na.all=TRUE)
  cor12 <- outer(1:ncol(a1dc), 1:ncol(a2dc),
                 Vectorize(function(i, j) corf(a1dc[,i], a2dc[,j])))
  weighted.mean(cor12, p12)
}

  # two continuous attributes
discor(weatherc$temperature, weatherc$humidity)
  # one discrete and one continuous attribute
discor(weatherc$outlook, weatherc$temperature)
discor(weatherc$temperature, weatherc$play)
  # two discrete attributes
discor(weatherc$outlook, weatherc$play)
  # attributes with missing values
discor(Soybean$seed, Soybean$roots)
##############
#19-4-3.R
##############
## subset evaluation for correlation-based filters
cfs.eval <- function(subset, target, data, cormat)
{
  if (length(subset <- unique(subset))>0)
  {
      cor.at <- mean(sapply(subset, function(a) cormat[a,target]))
      cor.aa <- mean(outer(subset, subset,
                           Vectorize(function(a1, a2)
                                     ifelse(a1!=a2, cormat[a1,a2], NA))),
                     na.rm=TRUE)
      cor.aa <- ifelse(is.finite(cor.aa), cor.aa, 0)
      length(subset)*cor.at/
        sqrt(length(subset)+length(subset)*(length(subset)-1)*cor.aa)
  }
  else
    -Inf
}

## correlation-based filter
cfs.filter <- function(formula, data, corf=symunc,
                       searchf=asel.search.greedy,
                       initf=asel.init.all, nextf=asel.next.backward)
{
  target <- y.var(formula)
  attributes <- x.vars(formula, data)
  atnames <- c(attributes, target)

  cormat <- outer(1:length(atnames), 1:length(atnames),
                  Vectorize(function(i, j)
                            ifelse(j<i, corf(data[[atnames[i]]], data[[atnames[j]]]),
                                   NA)))
  cormat[upper.tri(cormat)] <- t(cormat)[t(lower.tri(cormat))]
  dimnames(cormat) <- list(atnames, atnames)

  searchf(attributes, target,
          evalf=function(subset, target) cfs.eval(subset, target, data, cormat),
          initf=initf, nextf=nextf)
}

  # correlation-based filter for the weather data
cfs.filter(play~., weather)
cfs.filter(play~., weather, initf=asel.init.none, nextf=asel.next.forward)
cfs.filter(play~., weather, corf=discor)
cfs.filter(play~., weather, corf=discor,
           initf=asel.init.none, nextf=asel.next.forward)

  # correlation-based filter for the weatherc data
cfs.filter(play~., discnm.eqfreq(~., weatherc, 4))
cfs.filter(play~., weatherc, corf=discor)
cfs.filter(play~., weatherc, corf=discor,
           initf=asel.init.none, nextf=asel.next.forward)

  # correlation-based filter for the weatherr data
cfs.filter(playability~., weatherr, corf=discor)
cfs.filter(playability~., weatherr, corf=discor,
           initf=asel.init.none, nextf=asel.next.forward)

  # correlation-based filter for the Vehicle Silhouettes data
v.sel.cfs <- cfs.filter(Class~., discnm.eqfreq(~., v.train, 7))$subset
  # correlation-based filter for the Soybean data
s.sel.cfs <- cfs.filter(Class~., s.train)$subset
  # correlation-based filter for the Boston Housing data
bh.sel.cfs <- cfs.filter(medv~., discnm.eqfreq(~., bh.train, 7))$subset
##############
#19-4-4.R
##############
## subset evaluation for consistency-based filters
cons.eval <- function(subset, target, data)
{
  if (require(digest, quietly=TRUE))
    hashfun <- function(x) digest(as.numeric(x))
  else
    hashfun <- function(x) paste(as.numeric(x), collapse="")
  aind <- names(data) %in% subset
  datahash <- sapply(1:nrow(data), function(j) hashfun(data[j,aind]))
  -sum(sapply(unique(datahash),
              function(xh)
              sum(datahash==xh)/nrow(data)*entropy(data[datahash==xh,target])))
}

## consistency-based filter
cons.filter <- function(formula, data,
                        searchf=asel.search.greedy,
                        initf=asel.init.all, nextf=asel.next.backward)
{
  target <- y.var(formula)
  attributes <- x.vars(formula, data)
  searchf(attributes, target,
          evalf=function(subset, target) cons.eval(subset, target, data),
          initf=initf, nextf=nextf)
}

  # consistency-based filter for the weather data
cons.filter(play~., weather)
cons.filter(play~., weather, initf=asel.init.none, nextf=asel.next.forward)

  # consistency-based filter for the weatherc data
cons.filter(play~., discnm.eqfreq(~., weatherc, 4))
cons.filter(play~., discnm.eqfreq(~., weatherc, 4),
            initf=asel.init.none, nextf=asel.next.forward)
cons.filter(play~., weatherc)
cons.filter(play~., weatherc, initf=asel.init.none, nextf=asel.next.forward)

  # correlation-based filter for the weatherr data
cons.filter(playability~., discnm.eqfreq(~., weatherr, 4))
cons.filter(playability~., discnm.eqfreq(~., weatherr, 4),
            initf=asel.init.none, nextf=asel.next.forward)

  # consistency-based for the Vehicle Silhouettes data
v.sel.cons <- cons.filter(Class~., discnm.eqfreq(~., v.train, 7))$subset
  # consistency-based for the Soybean data
s.sel.cons <- cons.filter(Class~., s.train)$subset
  # consistency-based for the Boston Housing data
bh.sel.cons <- cons.filter(medv~., discnm.eqfreq(~., bh.train, 7))$subset
##############
#19-4-5.R
##############
## RELIEF dissimilarity
relief.diss <- function(x1, x2, rngs)
{
  ifelse(is.na(rd <- mapply(function(v1, v2, r)
                            ifelse(is.numeric(v1), abs(v1-v2)/r, v1!=v2),
                            x1, x2, rngs)),
         0, rd)
}

## RELIEF filter
relief.filter <- function(formula, data, k=1, K=floor(0.1*nrow(data)))
{
  attributes <- x.vars(formula, data)
  class <- y.var(formula)
  aind <- names(data) %in% attributes
  rngs <- ranges(data)[aind]

  util <- sapply(attributes, function(a) 0)

  for (i in 1:K)
  {
    xi <- sample(nrow(data), 1)
    x <- data[xi, ]
    data.x <- data[-xi,]
    hits <- arg.min((1:nrow(data.x))[data.x[[class]]==x[[class]]],
                    function(j) sum(relief.diss(data.x[j,aind], x[aind], rngs)),
                    k=k)
    misses <- arg.min((1:nrow(data.x))[data.x[[class]]!=x[[class]]],
                      function(j) sum(relief.diss(data.x[j,aind], x[aind], rngs)),
                      k=k)
    util <- util +
      rowSums(sapply(misses,
                     function(j) relief.diss(data.x[j,aind], x[aind], rngs)))/K-
        rowSums(sapply(hits,
                       function(j) relief.diss(data.x[j,aind], x[aind], rngs)))/K
  }
  sort(util, decreasing=TRUE)
}

  # RELIEF for the weather data
relief.filter(play~., weather, K=100)
relief.filter(play~., weather, k=3, K=100)

  # RELIEF for the weatherc data
relief.filter(play~., weatherc, K=100)
relief.filter(play~., weatherc, k=3, K=100)

  # RELIEF for the Vehicle Silhouettes data
v.utl.rel <- relief.filter(Class~., v.train, k=3, K=200)
  # RELIEF for the Soybean data
s.utl.rel <- relief.filter(Class~., s.train, k=3, K=200)
##############
#19-4-6.R
##############
## regression RELIEF filter
rrelief.filter <- function(formula, data, k=1, K=floor(0.1*nrow(data)), sigma=10)
{
  attributes <- x.vars(formula, data)
  target <- y.var(formula)
  aind <- names(data) %in% attributes
  rngs <- ranges(data)[aind]
  data <- predict.nrm(nrm.all(make.formula("", target), data), data)

  delta.f <- 0
  delta.i <- delta.fi <- sapply(attributes, function(a) 0)

  for (i in 1:K)
  {
    xi <- sample(nrow(data), 1)
    x <- data[xi, ]
    data.x <- data[-xi,]
    diss <- sapply(1:nrow(data.x),
                   function(j) sum(relief.diss(data.x[j,aind], x[aind], rngs)))
    neighbors <- arg.min(1:nrow(data.x), function(j) diss[j], k=k)
    nbranks <- rank(diss[neighbors])
    weights <- (weights <- exp(-(nbranks/sigma)^2))/sum(weights)
    adiff <- sapply(neighbors, function(j)
                               relief.diss(data.x[j,aind], x[aind], rngs))
    fdiff <- abs(data.x[neighbors,target]-x[,target])
    delta.f <- delta.f + sum(fdiff*weights)
    delta.i <- delta.i + rowSums(adiff %*% diag(weights, nrow=length(weights)))
    delta.fi <- delta.fi +
                  rowSums(adiff %*% diag(fdiff*weights, nrow=length(weights)))
  }
  sort(util<-delta.fi/delta.f-(delta.i-delta.fi)/(K-delta.f), decreasing=TRUE)
}

  # RELIEF for the weatherr data
rrelief.filter(playability~., weatherr, K=100)
rrelief.filter(playability~., weatherr, k=3, K=100)

  # RELIEF for the Boston Housing data
bh.utl.rel <- rrelief.filter(medv~., bh.train, k=3, K=200)
##############
#19-4-7.R
##############
## random forest filter
rf.filter <- function(formula, data, ...)
{
  if (require(randomForest, quietly=TRUE))
  {
    rf <- randomForest(formula, na.roughfix(data), importance=TRUE, ...)
    sort(importance(rf, type=1)[,1], decreasing=TRUE)

  }
  else
  {
    attributes <- x.vars(formula, data)
    names<-(rep(1, length(attributes), attributes))
  }
}

  # random forest filter for the weather data
rf.filter(play~., weather)
  # random forest filter for the weatherc data
rf.filter(play~., weatherc)
  # random forest filter for the weatherr data
rf.filter(playability~., weatherr)

  # random forest filter for the Vehicle Silhouettes data
v.utl.rf <- rf.filter(Class~., v.train)
  # random forest filter for the Soybean data
s.utl.rf <- rf.filter(Class~., s.train)
  # random forest filter for the BostonHousing data
bh.utl.rf <- rf.filter(medv~., bh.train)
##############
#19-4-8.R
##############
## cutoff based on decreasingly sorted named attribute utilities
cutoff <- function(utils, k=NA, p=NA)
{
  k <- ifelse(is.na(k), round(p*length(utils)), k)
  k <- ifelse(is.na(k), which.max(-diff(utils)), k)
  k <- ifelse(is.na(k), 1, k)

  names(utils)[1:min(k, length(utils))]
}

  # cutoff based on the random forest filter for the weather data
cutoff(rf.filter(play~., weather), k=3)
  # cutoff based on the random forest filter for the weatherc data
cutoff(rf.filter(play~., weatherc), p=0.5)
  # cutoff based on the random forest filter for the weatherr data
cutoff(rf.filter(playability~., weatherr))

  # cutoff for the Vehicle Silhouettes data
v.sel.simple <- cutoff(v.utl.simple, p=0.5)
v.sel.rf <- cutoff(v.utl.rf, p=0.5)
v.sel.rel <- cutoff(v.utl.rel, p=0.5)

  # cutoff for the Soybean data
s.sel.simple <- cutoff(s.utl.simple, p=0.5)
s.sel.rf <- cutoff(s.utl.rf, p=0.5)
s.sel.rel <- cutoff(s.utl.rel, p=0.5)

  # cutoff for the Boston Housing data
bh.sel.simple <- cutoff(bh.utl.simple, p=0.5)
bh.sel.rf <- cutoff(bh.utl.rf, p=0.5)
bh.sel.rel <- cutoff(bh.utl.rel, p=0.5)
##############
#19-4-9.R
##############
## filter-driven attribute selection search
asel.search.filter <- function(attributes, target, utils, evalf, penalty=0.01)
{
  ev <- function(subset)
  {
    ifelse(is.finite(v<-evalf(subset, target)), v-penalty*length(subset)*abs(v), v)
  }

  subsets <- unname(lapply(utils, function(u) names(utils)[utils>=u]))
  subsets.eval <- sapply(subsets, ev)
  s.best <- which.max(subsets.eval)

  list(subset=subsets[[s.best]], eval=subsets.eval[s.best])
}

  # filter attribute selection search for the weather data
  # using mutual information-based utility estimates
asel.search.filter(names(weather)[-5], "play",
                   simple.filter(play~., weather, dd=symunc),
                   evalf=function(subset, target) 1)
asel.search.filter(names(weather)[-5], "play",
                   simple.filter(play~., weather, dd=symunc),
                   evalf=function(subset, target) 1, penalty=0)
asel.search.filter(names(weather)[-5], "play",
                   simple.filter(play~., weather, dd=symunc),
                   evalf=function(subset, target) 1)
asel.search.filter(names(weather)[-5], "play",
                   simple.filter(play~., weather, dd=symunc),
                   evalf=function(subset, target) 1, penalty=0)
##############
#19-5-1.R
##############
## subset evaluation for attribute selection wrappers
wrapper.eval <- function(subset, target, data, alg, args=NULL, predf=predict,
                         perf=if (is.numeric(data[[target]])) mse else err,
                         evproc=crossval, evargs=list(k=5),
                         minrelsd=0.01, maxn=5)
{
  if (length(subset <- unique(subset))>0)
  {
    aind <- names(data) %in% subset
    ev <- do.call(evproc, c(list(alg, make.formula(target, subset), data,
                                 args=args, predf=predf),
                            evargs))
    p <- perf(ev$pred, ev$true)
    repeat
    {
      ev <- do.call(evproc, c(list(alg, make.formula(target, subset), data,
                                   args=args, predf=predf),
                              evargs))
      p <- c(p, perf(ev$pred, ev$true))
      if (length(p)>=maxn || sd(p)/abs(mean(p))<=minrelsd)
        break;
    }
    -mean(p)
  }
  else
    -Inf
}

  # wrapper evaluation for the weather data
wrapper.eval(c("outlook", "temperature"), "play", weather,
             rpart, args=list(minsplit=2),
             predf=function(...) predict(..., type="c"))
wrapper.eval(c("outlook", "temperature", "humidity"), "play", weather,
             rpart, args=list(minsplit=2),
             predf=function(...) predict(..., type="c"))
wrapper.eval(names(weather)[-5], "play", weather,
             rpart, args=list(minsplit=2),
             predf=function(...) predict(..., type="c"))

  # wrapper evaluation for the weatherc data
wrapper.eval(c("outlook", "temperature"), "play", weatherc,
             rpart, args=list(minsplit=2),
             predf=function(...) predict(..., type="c"))
wrapper.eval(c("outlook", "temperature", "humidity"), "play", weatherc,
             rpart, args=list(minsplit=2),
             predf=function(...) predict(..., type="c"))
wrapper.eval(names(weatherc)[-5], "play", weatherc,
             rpart, args=list(minsplit=2),
             predf=function(...) predict(..., type="c"))

  # wrapper evaluation for the weatherr data
wrapper.eval(c("outlook", "temperature"), "playability", weatherr,
             rpart, args=list(minsplit=2))
wrapper.eval(c("outlook", "temperature", "humidity"), "playability", weatherr,
             rpart, args=list(minsplit=2))
wrapper.eval(names(weatherr)[-5], "playability", weatherr,
             rpart, args=list(minsplit=2))
##############
#19-5-2.R
##############
## wrapper attribute selection
wrapper.select <- function(formula, data, alg, args=NULL, predf=predict,
                           searchf=asel.search.greedy,
                           initf=asel.init.all, nextf=asel.next.backward,
                           perf=if (is.numeric(data[[target]])) mse else err, ...)
{
  target <- y.var(formula)
  attributes <- x.vars(formula, data)
  searchf(attributes, target,
          evalf=function(subset, target)
                wrapper.eval(subset, target, data, alg, args, predf, perf, ...),
          initf=initf, nextf=nextf)
}

  # wrapper selection for the weather data
wrapper.select(play~., weather, rpart, args=list(minsplit=2),
               predf=function(...) predict(..., type="c"))
wrapper.select(play~., weather, rpart, args=list(minsplit=2),
               predf=function(...) predict(..., type="c"),
               initf=asel.init.none, nextf=asel.next.forward)

  # wrapper selection for the weatherc data
wrapper.select(play~., weatherc, rpart, args=list(minsplit=2),
               predf=function(...) predict(..., type="c"))
wrapper.select(play~., weatherc, rpart, args=list(minsplit=2),
               predf=function(...) predict(..., type="c"),
               initf=asel.init.none, nextf=asel.next.forward)

  # wrapper selection for the weatherr data
wrapper.select(playability~., weatherr, rpart, args=list(minsplit=2))
wrapper.select(playability~., weatherr, rpart, args=list(minsplit=2),
               initf=asel.init.none, nextf=asel.next.forward)

  # wrapper selection for the Vehicle Silhouettes data
v.sel.fwd <- wrapper.select(Class~., v.train, rpart,
                            predf=function(...) predict(..., type="c"),
                            initf=asel.init.none, nextf=asel.next.forward)
v.sel.bwd <- wrapper.select(Class~., v.train, rpart,
                            predf=function(...) predict(..., type="c"))

  # wrapper selection for the Soybean data
s.sel.fwd <- wrapper.select(Class~., s.train, rpart,
                            predf=function(...) predict(..., type="c"),
                            initf=asel.init.none, nextf=asel.next.forward)
s.sel.bwd <- wrapper.select(Class~., s.train, rpart,
                            predf=function(...) predict(..., type="c"))

  # wrapper selection for the Boston Housing data
bh.sel.fwd <- wrapper.select(medv~., bh.train, rpart,
                             initf=asel.init.none, nextf=asel.next.forward)
bh.sel.bwd <- wrapper.select(medv~., bh.train, rpart)
##############
#19-5-3.R
##############
## filter-driven wrapper attribute selection
wrapper.filter.select <- function(formula, data, utils, alg, args=NULL,
                                  predf=predict, searchf=asel.search.filter,
                                  perf=if (is.numeric(data[[target]])) mse else err,
                                  ...)
{
  target <- y.var(formula)
  attributes <- x.vars(formula, data)
  searchf(attributes, target, utils,
          evalf=function(subset, target)
                wrapper.eval(subset, target, data, alg, args, predf, perf, ...))
}

  # simple filter-driven wrapper selection for the weather data
wrapper.filter.select(play~., weather, simple.filter(play~., weather),
                      rpart, args=list(minsplit=2),
                      predf=function(...) predict(..., type="c"))

  # simple filter-driven wrapper selection for the weatherc data
wrapper.filter.select(play~., weatherc, simple.filter(play~., weatherc),
                      rpart, args=list(minsplit=2),
                      predf=function(...) predict(..., type="c"))

  # simple filter-driven wrapper selection for the weatherr data
wrapper.filter.select(playability~., weatherr,
                      simple.filter(playability~., weatherr),
                      rpart, args=list(minsplit=2))

  # RF filter-driven wrapper selection for the Vehicle Silhouettes data
v.sel.flt <- wrapper.filter.select(Class~., Vehicle, rf.filter(Class~., v.train),
                                   rpart, predf=function(...) predict(..., type="c"))

  # RF filter-driven wrapper selection for the Soybean data
s.sel.flt <- wrapper.filter.select(Class~., Soybean, rf.filter(Class~., s.train),
                                   rpart, predf=function(...) predict(..., type="c"))

  # RF filter-driven wrapper selection for the Boston Housing data
bh.sel.flt <- wrapper.filter.select(medv~., BostonHousing,
                                    rf.filter(medv~., bh.train), rpart)
##############
#19-6-1.R
##############
  # selected attribute subsets
v.self <- list(nosel=setdiff(names(v.train), "Class"),
               simple=v.sel.simple, cfs=v.sel.cfs, cons=v.sel.cons,
               rel=v.sel.rel, rf=v.sel.rf)
s.self <- list(nosel=setdiff(names(s.train), "Class"),
               simple=s.sel.simple, cfs=s.sel.cfs, cons=s.sel.cons,
               rel=s.sel.rel, rf=s.sel.rf)
bh.self <- list(nosel=setdiff(names(bh.train), "medv"),
                simple=bh.sel.simple, cfs=bh.sel.cfs, cons=bh.sel.cons,
                rel=bh.sel.rel, rf=bh.sel.rf)

  # misclassification error
v.errf <- sapply(v.self, function(sel)
                         {
                           tree <- rpart(make.formula("Class", sel), v.train)
                           err(predict(tree, v.test, type="c"), v.test$Class)
                         })
s.errf <- sapply(s.self, function(sel)
                         {
                           tree <- rpart(make.formula("Class", sel), s.train)
                           err(predict(tree, s.test, type="c"), s.test$Class)
                         })
bh.errf <- sapply(bh.self, function(sel)
                           {
                             tree <- rpart(make.formula("medv", sel), bh.train)
                             mse(predict(tree, bh.test), bh.test$medv)
                           })

  # attribute subset size
v.sizef <- sapply(v.self, length)
s.sizef <- sapply(s.self, length)
bh.sizef <- sapply(bh.self, length)

barplot(v.errf, main="Vehicle Silhouettes", ylab="Error", las=2)
lines(c(0, 8), rep(v.errf[1], 2), lty=2)
barplot(v.sizef, main="Vehicle Silhouettes", ylab="Size", las=2)

barplot(s.errf, main="Soybean", ylab="Error", las=2)
lines(c(0, 8), rep(s.errf[1], 2), lty=2)
barplot(s.sizef, main="Soybean", ylab="Size", las=2)

barplot(bh.errf, main="Boston Housing", ylab="Error", las=2)
lines(c(0, 8), rep(bh.errf[1], 2), lty=2)
barplot(bh.sizef, main="Boston Housing", ylab="Size", las=2)
##############
#19-6-2.R
##############
  # selected attribute subsets
v.selw <- list(nosel=setdiff(names(v.train), "Class"),
               fwd=v.sel.fwd$subset, bwd=v.sel.bwd$subset, flt=v.sel.flt$subset)
s.selw <- list(nosel=setdiff(names(s.train), "Class"),
               fwd=s.sel.fwd$subset, bwd=s.sel.bwd$subset, flt=s.sel.flt$subset)
bh.selw <- list(nosel=setdiff(names(bh.train), "medv"),
                fwd=bh.sel.fwd$subset, bwd=bh.sel.bwd$subset, flt=bh.sel.flt$subset)

  # misclassification error
v.errw <- sapply(v.selw, function(sel)
                         {
                           tree <- rpart(make.formula("Class", sel), v.train)
                           err(predict(tree, v.test, type="c"), v.test$Class)
                         })
s.errw <- sapply(s.selw, function(sel)
                         {
                           tree <- rpart(make.formula("Class", sel), s.train)
                           err(predict(tree, s.test, type="c"), s.test$Class)
                         })
bh.errw <- sapply(bh.selw, function(sel)
                           {
                             tree <- rpart(make.formula("medv", sel), bh.train)
                             mse(predict(tree, bh.test), bh.test$medv)
                           })

  # attribute subset size
v.sizew <- sapply(v.selw, length)
s.sizew <- sapply(s.selw, length)
bh.sizew <- sapply(bh.selw, length)

barplot(v.errw, main="Vehicle Silhouettes", ylab="Error", las=2)
lines(c(0, 5), rep(v.errw[1], 2), lty=2)
barplot(v.sizew, main="Vehicle Silhouettes", ylab="Size", las=2)

barplot(s.errw, main="Soybean", ylab="Error", las=2)
lines(c(0, 5), rep(s.errw[1], 2), lty=2)
barplot(s.sizew, main="Soybean", ylab="Size", las=2)

barplot(bh.errw, main="Boston Housing", ylab="Error", las=2)
lines(c(0, 5), rep(bh.errw[1], 2), lty=2)
barplot(bh.sizew, main="Boston Housing", ylab="Size", las=2)
##############
#20-1.R
##############
## complexity parameter corresponding to the minimum cross-validated error
cpmin <- function(cptab) { cptab[which.min(cptab[,4])[1], 1] }

## complexity parameter corresponding to the smallest tree within 1-SD
## from the minimum cross-validated error
cp1sd <- function(cptab)
{ cptab[which(cptab[,4]<min(cptab[,4]) + cptab[which.min(cptab[,4]),5])[1], 1] }

## sequence of complexity parameter values corresponding to the minimum
## cross-validated error, s next smaller trees, and l next larger trees
cpminrange <- function(cptab, s=5, l=5)
{
  m <- which.min(cptab[,4])[1]
  cptab[max(m-s, 1):min(m+l, nrow(cptab)), 1]
}

## grow and prune an rpart tree using minimum-error cost-complexity pruning
rpart.pmin <- function(formula, data, ...)
{
  tree.f <- rpart(formula, data, minsplit=2, cp=0, ...)
  prune(tree.f, cpmin(tree.f$cptable))
}

## grow and prune an rpart tree using 1-SD cost-complexity pruning
rpart.p1sd <- function(formula, data, ...)
{
  tree.f <- rpart(formula, data, minsplit=2, cp=0, ...)
  prune(tree.f, cp1sd(tree.f$cptable))
}

## create a piecewise linear model represented by an rpart regression tree
## with linear models corresponding to leaves
lmrpart <- function(formula, data, skip.attr=FALSE, ...)
{
  m.tree <- rpart(formula, data, ...)
  m.leaves <- sort(unique(predict(m.tree, data)))
  lmattr <- if (skip.attr)
              setdiff(x.vars(formula, data), setdiff(m.tree$frame$var, "<leaf>"))
            else "."
  m.lm <- `names<-`(lapply(m.leaves, function(l)
                                     lm(make.formula(y.var(formula), lmattr),
                                        data[predict(m.tree, data)==l,])),
                    m.leaves)
  `class<-`(list(tree=m.tree, lm=m.lm), "lmrpart")
}

## prediction method for lmrpart
predict.lmrpart <- function(model, data)
{
  leaves <- as.character(predict(model$tree, data))
  sapply(1:nrow(data),
         function(i) predict(model$lm[[leaves[i]]], data[i,]))
}
##############
#20-2.R
##############
library(dmr.claseval)
library(dmr.util)
library(dmr.trans)

library(rpart)
library(rpart.plot)
library(randomForest)
#https://kdd.ics.uci.edu/databases/census-income/census-income.html
census <- read.table("./census-income.data",
                     sep=",", na.strings="?", strip.white=TRUE)
census.test <- read.table("./census-income.test",
                          sep=",", na.strings="?", strip.white=TRUE)
names(census) <- c("age",
                   "class.of.worker",
                   "detailed.industry.recode",
                   "detailed.occupation.recode",
                   "education",
                   "wage.per.hour",
                   "enroll.in.edu.inst.last.wk",
                   "marital.stat",
                   "major.industry.code",
                   "major.occupation.code",
                   "race",
                   "hispanic.origin",
                   "sex",
                   "member.of.a.labor.union",
                   "reason.for.unemployment",
                   "full.or.part.time.employment.stat",
                   "capital.gains",
                   "capital.losses",
                   "dividends.from.stocks",
                   "tax.filer.stat",
                   "region.of.previous.residence",
                   "state.of.previous.residence",
                   "detailed.household.and.family.stat",
                   "detailed.household.summary.in.household",
                   "instance.weight",
                   "migration.code.change.in.msa",
                   "migration.code.change.in.reg",
                   "migration.code.move.within.reg",
                   "live.in.this.house.1.year.ago",
                   "migration.prev.res.in.sunbelt",
                   "num.persons.worked.for.employer",
                   "family.members.under.18",
                   "country.of.birth.father",
                   "country.of.birth.mother",
                   "country.of.birth.self",
                   "citizenship",
                   "own.business.or.self.employed",
                   "fill.inc.questionnaire.for.veterans.admin",
                   "veterans.benefits",
                   "weeks.worked.in.year",
                   "year",
                   "income")
names(census.test) <- names(census)

ci.discrete <- c("detailed.industry.recode", "detailed.occupation.recode",
                 "own.business.or.self.employed", "veterans.benefits", "year")
for (a in ci.discrete)
{
  census[[a]] <- as.factor(census[[a]])
  census.test[[a]] <- as.factor(census.test[[a]])
}

census$instance.weight <- NULL
census.test$instance.weight <- NULL

ci.labels <- c("low", "high")
census$income <- factor(ifelse(census$income=="50000+.", "high", "low"),
                        levels=ci.labels)
census.test$income <- factor(ifelse(census.test$income=="50000+.", "high", "low"),
                             levels=ci.labels)

set.seed(12)

rci <- runif(nrow(census))
ci.train <- census[rci>=0.33,]
ci.val <- census[rci<0.33,]
ci.train.small <- census[rci>=0.9,]

ci.tree.d <- rpart(income~., ci.train)

ci.tree.d.pred <- predict(ci.tree.d, ci.val, type="c")
err(ci.tree.d.pred, ci.val$income)

ci.tree.d.cm <- confmat(ci.tree.d.pred, ci.val$income)

ci.tree.d.tpr <- tpr(ci.tree.d.cm)
ci.tree.d.fpr <- fpr(ci.tree.d.cm)

ci.tree.d.roc <- roc(predict(ci.tree.d, ci.val)[,2], ci.val$income)
plot(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=8)
auc(ci.tree.d.roc)

ci.tree.d.cut06 <- ci.tree.d.roc$cutoff[ci.tree.d.roc$tpr>0.6]

#error not found ci.tree.d.prob
ci.tree.d.cm06 <- confmat(cutclass(ci.tree.d.prob, ci.tree.d.cut06[1], ci.labels),
                          ci.val$income)
ci.tree.d.tpr06 <- tpr(ci.tree.d.cm06)
ci.tree.d.fpr06 <- fpr(ci.tree.d.cm06)
points(ci.tree.d.fpr06, ci.tree.d.tpr06, pch=1)

ci.cost2 <- matrix(c(0, 1, 2, 0), nrow=2, byrow=TRUE)
ci.tree.c2 <- rpart(income~., ci.train, parms=list(loss=ct.cost2))

ci.tree.c2.pred <- predict(ci.tree.c2, ci.val, type="c")
err(ci.tree.c2.pred, ci.val$income)
  # confusion matrix
ci.tree.c2.cm <- confmat(ci.tree.c2.pred, ci.val$income)
  # true positives/false negative rates
ci.tree.c2.tpr <- tpr(ci.tree.c2.cm)
ci.tree.c2.fpr <- fpr(ci.tree.c2.cm)

ci.tree.c2.prob <- predict(ci.tree.c2, ci.val)[,2]
ci.tree.c2.roc <- roc(ci.tree.c2.prob, ci.val$income)
plot(ci.tree.c2.roc$fpr, ci.tree.c2.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
points(ci.tree.c2.fpr, ci.tree.c2.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.c2.roc)

ci.tree.c2.cut07 <- ci.tree.c2.roc$cutoff[ci.tree.c2.roc$tpr>0.7]
ci.tree.c2.cm07 <- confmat(cutclass(ci.tree.c2.prob, ci.tree.c2.cut07[1], ci.labels),
                           ci.val$income)
ci.tree.c2.tpr07 <- tpr(ci.tree.c2.cm07)
ci.tree.c2.fpr07 <- fpr(ci.tree.c2.cm07)
points(ci.tree.c2.fpr07, ci.tree.c2.tpr07, pch=1)

ci.tree.w2 <- rpart(income~., census.train,
                    weights=ifelse(census.train$income=="high", 2, 1))

ci.cost5 <- matrix(c(0, 1, 5, 0), nrow=2, byrow=TRUE)
ci.tree.c5 <- rpart(income~., ci.train, parms=list(loss=ci.cost5))
  # error
ci.tree.c5.pred <- predict(ci.tree.c5, ci.val, type="c")
err(ci.tree.c5.pred, ci.val$income)
  # confusion matrix
ci.tree.c5.cm <- confmat(ci.tree.c5.pred, ci.val$income)
  # true positive/false positive rates
ci.tree.c5.tpr <- tpr(ci.tree.c5.cm)
ci.tree.c5.fpr <- fpr(ci.tree.c5.cm)
  # ROC
ci.tree.c5.prob <- predict(ci.tree.c5, ci.val)[,2]
ci.tree.c5.roc <- roc(ci.tree.c5.prob, ci.val$income)
plot(ci.tree.c5.roc$fpr, ci.tree.c5.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
points(ci.tree.c5.fpr, ci.tree.c5.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.c5.roc)

ci.tree.c5.cut08 <- ci.tree.c5.roc$cutoff[ci.tree.c5.roc$tpr>0.8]
ci.tree.c5.cm08 <- confmat(cutclass(ci.tree.c5.prob, ci.tree.c5.cut08[1], ci.labels),
                           ci.val$income)
ci.tree.c5.tpr08 <- tpr(ci.tree.c5.cm08)
ci.tree.c5.fpr08 <- fpr(ci.tree.c5.cm08)
points(ci.tree.c5.fpr08, ci.tree.c5.tpr08, pch=1)

ci.tree.c10.prob <- predict(ci.tree.c10, ci.val)[,2]
ci.tree.c10.roc <- roc(ci.tree.c10.prob, ci.val$income)
plot(ci.tree.c10.roc$fpr, ci.tree.c10.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate")
points(ci.tree.c10.fpr, ci.tree.c10.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.c10.roc)

ci.tree.f <- rpart(income~., ci.train, minsplit=2, cp=0)

  # minimum-error cost-complexity pruning
ci.tree.pmin <- prune(ci.tree.f, cpmin(ci.tree.f$cptable))
  # 1-sd cost-complexity pruning
ci.tree.p1sd <- prune(ci.tree.f, cp1sd(ci.tree.f$cptable))

c(default=nrow(ci.tree.d$frame), full=nrow(ci.tree.f$frame),
  pruned.min=nrow(ci.tree.pmin$frame), pruned.1sd=nrow(ci.tree.p1sd$frame))

ci.tree.f.pred <- predict(ci.tree.f, ci.val, type="c")
err(ci.tree.f.pred, ci.val$income)

ci.tree.f.cm <- confmat(ci.tree.f.pred, ci.val$income)
ci.tree.f.tpr <- tpr(ci.tree.f.cm)
ci.tree.f.fpr <- fpr(ci.tree.f.cm)

ci.tree.f.prob <- predict(ci.tree.f, ci.val)[,2]
ci.tree.f.roc <- roc(ci.tree.f.prob, ci.val$income)
plot(ci.tree.f.roc$fpr, ci.tree.f.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
points(ci.tree.f.fpr, ci.tree.f.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.f.roc)

ci.tree.pmin.pred <- predict(ci.tree.pmin, ci.val, type="c")
err(ci.tree.pmin.pred, ci.val$income)

ci.tree.pmin.cm <- confmat(ci.tree.pmin.pred, ci.val$income)
ci.tree.pmin.tpr <- tpr(ci.tree.pmin.cm)
ci.tree.pmin.fpr <- fpr(ci.tree.pmin.cm)

ci.tree.pmin.prob <- predict(ci.tree.pmin, ci.val)[,2]
ci.tree.pmin.roc <- roc(ci.tree.pmin.prob, ci.val$income)
plot(ci.tree.pmin.roc$fpr, ci.tree.pmin.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="Minimum error CCP")
points(ci.tree.pmin.fpr, ci.tree.pmin.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.pmin.roc)

ci.tree.p1sd.pred <- predict(ci.tree.p1sd, ci.val, type="c")
err(ci.tree.p1sd.pred, ci.val$income)

ci.tree.p1sd.cm <- confmat(ci.tree.p1sd.pred, ci.val$income)
ci.tree.p1sd.tpr <- tpr(ci.tree.p1sd.cm)
ci.tree.p1sd.fpr <- fpr(ci.tree.p1sd.cm)

ci.tree.p1sd.prob <- predict(ci.tree.p1sd, ci.val)[,2]
ci.tree.p1sd.roc <- roc(ci.tree.p1sd.prob, ci.val$income)
plot(ci.tree.p1sd.roc$fpr, ci.tree.p1sd.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="1-SD CCP")
points(ci.tree.p1sd.fpr, ci.tree.p1sd.tpr, pch=8)
lines(ci.tree.d.roc$fpr, ci.tree.d.roc$tpr, lty=2)
points(ci.tree.d.fpr, ci.tree.d.tpr, pch=4)
auc(ci.tree.p1sd.roc)

ci.tree.p1sd.cut08 <- ci.tree.p1sd.roc$cutoff[ci.tree.p1sd.roc$tpr>0.8]
ci.tree.p1sd.cm08 <- confmat(cutclass(ci.tree.p1sd.prob, ci.tree.p1sd.cut08[1],
                                      ci.labels),
                             ci.val$income)
ci.tree.p1sd.tpr08 <- tpr(ci.tree.p1sd.cm08)
ci.tree.p1sd.fpr08 <- fpr(ci.tree.p1sd.cm08)
points(ci.tree.p1sd.fpr08, ci.tree.p1sd.tpr08, pch=1)

ci.tree.c5f <- rpart(income~., ci.train, minsplit=2, cp=0, parms=list(loss=ci.cost5))

ci.tree.c5f.pred <- predict(ci.tree.c5f, ci.val, type="c")
err(ci.tree.c5f.pred, ci.val$income)

ci.tree.c5f.cm <- confmat(ci.tree.c5f.pred, ci.val$income)
ci.tree.c5f.tpr <- tpr(ci.tree.c5f.cm)
ci.tree.c5f.fpr <- fpr(ci.tree.c5f.cm)

ci.tree.c5f.prob <- predict(ci.tree.c5f, ci.val)[,2]
ci.tree.c5f.roc <- roc(ci.tree.c5f.prob, ci.val$income)
plot(ci.tree.c5f.roc$fpr, ci.tree.c5f.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate")
points(ci.tree.c5f.fpr, ci.tree.c5f.tpr, pch=8)
lines(ci.tree.c5.roc$fpr, ci.tree.c5.roc$tpr, lty=2)
points(ci.tree.c5.fpr, ci.tree.c5.tpr, pch=4)
auc(ci.tree.c5f.roc)

  # minimum-error cost-complexity pruning (with cp determined based on ci.tree.f)
ci.tree.c5pmin <- prune(ci.tree.c5f, cpmin(ci.tree.f$cptable))
  # 1-sd cost-complexity pruning  (with cp terming based on ci.tree.f)
ci.tree.c5p1sd <- prune(ci.tree.c5f, cp1sd(ci.tree.f$cptable))

ci.tree.c5pmin.pred <- predict(ci.tree.c5pmin, ci.val, type="c")
err(ci.tree.c5pmin.pred, ci.val$income)

ci.tree.c5pmin.cm <- confmat(ci.tree.c5pmin.pred, ci.val$income)
ci.tree.c5pmin.tpr <- tpr(ci.tree.c5pmin.cm)
ci.tree.c5pmin.fpr <- fpr(ci.tree.c5pmin.cm)

ci.tree.c5pmin.prob <- predict(ci.tree.c5pmin, ci.val)[,2]
ci.tree.c5pmin.roc <- roc(ci.tree.c5pmin.prob, ci.val$income)
plot(ci.tree.c5pmin.roc$fpr, ci.tree.c5pmin.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="Minimum error CCP")
points(ci.tree.c5pmin.fpr, ci.tree.c5pmin.tpr, pch=8)
lines(ci.tree.c5.roc$fpr, ci.tree.c5.roc$tpr, lty=2)
points(ci.tree.c5.fpr, ci.tree.c5.tpr, pch=4)
auc(ci.tree.c5pmin.roc)

ci.tree.c5p1sd.pred <- predict(ci.tree.c5p1sd, ci.val, type="c")
err(ci.tree.c5p1sd.pred, ci.val$income)

ci.tree.c5p1sd.cm <- confmat(ci.tree.c5p1sd.pred, ci.val$income)
ci.tree.c5p1sd.tpr <- tpr(ci.tree.c5p1sd.cm)
ci.tree.c5p1sd.fpr <- fpr(ci.tree.c5p1sd.cm)

ci.tree.c5p1sd.prob <- predict(ci.tree.c5p1sd, ci.val)[,2]
ci.tree.c5p1sd.roc <- roc(ci.tree.c5p1sd.prob, ci.val$income)
plot(ci.tree.c5p1sd.roc$fpr, ci.tree.c5p1sd.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="1-SD CCP")
points(ci.tree.c5p1sd.fpr, ci.tree.c5p1sd.tpr, pch=8)
lines(ci.tree.c5.roc$fpr, ci.tree.c5.roc$tpr, lty=2)
points(ci.tree.c5.fpr, ci.tree.c5.tpr, pch=4)
auc(ci.tree.c5p1sd.roc)

ci.tree.c5p1sd.cut085 <- ci.tree.c5p1sd.roc$cutoff[ci.tree.c5p1sd.roc$tpr>0.85]
ci.tree.c5p1sd.cm085 <- confmat(cutclass(ci.tree.c5p1sd.prob,
                                         ci.tree.c5p1sd.cut085[1], ci.labels),
                                ci.val$income)
ci.tree.c5p1sd.tpr085 <- tpr(ci.tree.c5p1sd.cm085)
ci.tree.c5p1sd.fpr085 <- fpr(ci.tree.c5p1sd.cm085)
points(ci.tree.c5p1sd.fpr085, ci.tree.c5p1sd.tpr085, pch=1)

  # aggregation (ensure no more than 32 discrete attribute values)
ci.aggm <- agg.all(income~., ci.train.small, 31)
cirf.train <- predict.agg(ci.aggm, ci.train.small)
cirf.val <- predict.agg(ci.aggm, ci.val)
  # imputation (ensure no missing values)
cirf.impm <- imp.all(income~., cirf.train)
cirf.train <- predict.imp(cirf.impm, cirf.train)
cirf.val <- predict.imp(cirf.impm, cirf.val)

ci.rf <- randomForest(income~., cirf.train, importance=TRUE)

ci.rf.pred <- predict(ci.rf, cirf.val)
err(ci.rf.pred, cirf.val$income)
ci.rf.cm <- confmat(ci.rf.pred, cirf.val$income)

ci.rf.tpr <- tpr(ci.rf.cm)
ci.rf.fpr <- fpr(ci.rf.cm)

ci.rf.prob <- predict(ci.rf, cirf.val, type="p")[,2]
ci.rf.roc <- roc(ci.rf.prob, cirf.val$income)
plot(ci.rf.roc$fpr, ci.rf.roc$tpr, type="l", xlab="FP rate", ylab="TP rate")
points(ci.rf.fpr, ci.rf.tpr, pch=8)
auc(ci.rf.roc)

ci.rf.cut09 <- ci.rf.roc$cutoff[ci.rf.roc$tpr>0.9]
ci.rf.cm09 <- confmat(cutclass(ci.rf.prob, ci.rf.cut09[1], ci.labels), ci.val$income)
ci.rf.tpr09 <- tpr(ci.rf.cm09)
ci.rf.fpr09 <- fpr(ci.rf.cm09)
points(ci.rf.fpr09, ci.rf.tpr09, pch=1)

varImpPlot(ci.rf, type=1)

ci.attr.utl <- sort(importance(ci.rf)[,1], decreasing=TRUE)
ci.asets <-
  `names<-`(lapply(c(10, 25, 50, 100),
                   function(p)
                   names(ci.attr.utl)[1:round(p*length(ci.attr.utl)/100)]),
            paste("as", c(10, 25, 50, 100), "p", sep=""))

  # models using selected subsets
ci.tree.c5.as <-
  lapply(ci.attrs,
         function(as)
         {
           tree.as <- rpart(make.formula("income", as), ci.train,
                            parms=list(loss=ci.cost5))
           cm.as <- confmat(predict(tree.as, ci.val, type="c"), ci.val$income)
           roc.as <- roc(predict(tree.as, ci.val)[,2], ci.val$income)
           list(tree=tree.as,
                tpr=tpr(cm.as),
                fpr=fpr(cm.as),
                roc=roc.as,
                auc=auc(roc.as))
         })

sapply(ci.tree.c5.as, function(ta) ta$auc)

  # default operating point
ci.tree.c5p1sd.test.pred <- predict(ci.tree.c5p1sd, census.test, type="c")
ci.tree.c5p1sd.test.cm <- confmat(ci.tree.c5p1sd.test.pred, census.test$income)
ci.tree.c5p1sd.test.tpr <- tpr(ci.tree.c5p1sd.test.cm)
ci.tree.c5p1sd.test.fpr <- fpr(ci.tree.c5p1sd.test.cm)
  # ROC
ci.tree.c5p1sd.test.prob <- predict(ci.tree.c5p1sd, census.test)[,2]
ci.tree.c5p1sd.test.roc <- roc(ci.tree.c5p1sd.test.prob, census.test$income)
plot(ci.tree.c5p1sd.test.roc$fpr, ci.tree.c5p1sd.test.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="Decision tree")
points(ci.tree.c5p1sd.test.fpr, ci.tree.c5p1sd.test.tpr, pch=8)
auc(ci.tree.c5p1sd.roc)
  # operating point shifted based on the validation set
ci.tree.c5p1sd.test.cm085 <- confmat(cutclass(ci.tree.c5p1sd.test.prob,
                                              ci.tree.c5p1sd.cut085[1], ci.labels),
                                     census.test$income)
ci.tree.c5p1sd.test.tpr085 <- tpr(ci.tree.c5p1sd.test.cm085)
ci.tree.c5p1sd.test.fpr085 <- fpr(ci.tree.c5p1sd.test.cm085)
points(ci.tree.c5p1sd.test.fpr085, ci.tree.c5p1sd.test.tpr085, pch=1)

  # test set preprocessing
cirf.test <- predict.agg(ci.aggm, census.test)
cirf.test <- predict.imp(cirf.impm, cirf.test)
  # default operating point
ci.rf.test.pred <- predict(ci.rf, cirf.test)
ci.rf.test.cm <- confmat(ci.rf.test.pred, cirf.test$income)
ci.rf.test.tpr <- tpr(ci.rf.test.cm)
ci.rf.test.fpr <- fpr(ci.rf.test.cm)
  # ROC
ci.rf.test.prob <- predict(ci.rf, cirf.test, type="p")[,2]
ci.rf.test.roc <- roc(ci.rf.test.prob, cirf.test$income)
plot(ci.rf.test.roc$fpr, ci.rf.test.roc$tpr, type="l",
     xlab="FP rate", ylab="TP rate", main="Random forest")
points(ci.rf.test.fpr, ci.rf.test.tpr, pch=8)
auc(ci.rf.test.roc)
  # operating point shifted based on the validation set
ci.rf.test.cm09 <- confmat(cutclass(ci.rf.test.prob, ci.rf.cut09[1], ci.labels),
                           cirf.test$income)
ci.rf.test.tpr09 <- tpr(ci.rf.test.cm09)
ci.rf.test.fpr09 <- fpr(ci.rf.test.cm09)
points(ci.rf.test.fpr09, ci.rf.test.tpr09, pch=1)

prp(prune(ci.tree.c5p1sd, 0.01), varlen=8, faclen=2)
##############
#20-3.R
##############
library(dmr.regeval)
library(dmr.util)
library(dmr.trans)

library(rpart)
library(rpart.plot)
library(randomForest)

  # read column names (extracted from the data description web page)
commnorm.names <- read.table("../Data/commnorm.names",
                             stringsAsFactors=FALSE)[,1]
  # read the actual data
commnorm <- read.table("../Data/communities.data",
                       sep=",", na.strings="?", col.names=commnorm.names)
  # input attribute names
cn.input.attrs <- names(commnorm)[6:127]

set.seed(12)

rcn <- runif(nrow(commnorm))
cn.train <- commnorm[rcn>=0.33,]
cn.val <- commnorm[rcn<0.33,]

sum(complete.cases(cn.train))/nrow(cn.train)
sum(complete.cases(cn.val))/nrow(cn.val)

  # attributes with many (>50%) missing values
cn.input.attrs.miss <-
  names(which(sapply(cn.input.attrs,
                     function(a)
                     sum(is.na(cn.train[a]))/nrow(cn.train))>0.5))

  # attributes with many (>10%) outliers
cn.input.attrs.out <-
  names(which(sapply(cn.input.attrs,
                     function(a)
                     length(boxplot(cn.train[a], range=2, plot=FALSE)$out)/
                     nrow(cn.train))>0.1))

cn.input.attrs.cor <- cor(cn.train[,cn.input.attrs], use="pairwise.complete.obs")
cn.input.attrs.corind <- which(upper.tri(cn.input.attrs.cor) &
                               abs(cn.input.attrs.cor)>0.98, arr.ind=TRUE)
cn.input.attrs.corpairs <- data.frame(a1=cn.input.attrs[cn.input.attrs.corind[,1]],
                                      a2=cn.input.attrs[cn.input.attrs.corind[,2]])

cn.impm <- imp.all(make.formula(NULL, cn.input.attrs), cn.train)
cni.train <- predict.imp(cn.impm, cn.train)
cni.val <- predict.imp(cn.impm, cn.val)

cn.tree.d <- rpart(make.formula("ViolentCrimesPerPop", cn.input.attrs), cn.train)
r2(predict(cn.tree.d, cn.val), cn.val$ViolentCrimesPerPop)

  # fully-grown tree
cn.tree.f <- rpart(make.formula("ViolentCrimesPerPop", cn.input.attrs), cn.train,
                   minsplit=2, cp=0)
r2(predict(cn.tree.f, cn.val), cn.val$ViolentCrimesPerPop)
  # minimum-error cost-complexity pruning
cn.tree.pmin <- prune(cn.tree.f, cpmin(cn.tree.f$cptable))
r2(predict(cn.tree.pmin, cn.val), cn.val$ViolentCrimesPerPop)
  # 1-sd cost-complexity pruning
cn.tree.p1sd <- prune(cn.tree.f, cp1sd(cn.tree.f$cptable))
r2(predict(cn.tree.p1sd, cn.val), cn.val$ViolentCrimesPerPop)

  # 10x10-fold cross-validated R2 values for the most promising cp sequence
cn.cp.cv <-
  sapply(unname(cpminrange(cn.tree.f$cptable, 5, 10)),
         function(cp)
         {
           cv <- crossval(rpart, make.formula("ViolentCrimesPerPop", cn.input.attrs),
                          cn.train, args=list(cp=cp, minsplit=2, xval=0), n=10)
           `names<-`(r2(cv$pred, cv$true), cp)
         })

cn.tree.pcv <- prune(cn.tree.f, as.numeric(names(cn.cp.cv)[which.max(cn.cp.cv)]))
r2(predict(cn.tree.pcv, cn.val), cn.val$ViolentCrimesPerPop)

cn.lm <- lm(make.formula("ViolentCrimesPerPop", cn.input.attrs), cni.train)
r2(predict(cn.lm, cni.val), cni.val$ViolentCrimesPerPop)

signif.attrs <- cn.input.attrs[(summary(cn.lm)$coefficients)[-1,4]<0.05]
cn.lm.s <- lm(make.formula("ViolentCrimesPerPop", signif.attrs), cni.train)
r2(predict(cn.lm.s, cni.val), cni.val$ViolentCrimesPerPop)

cn.rf <- randomForest(make.formula("ViolentCrimesPerPop", cn.input.attrs), cni.train,
                      importance=TRUE)
r2(predict(cn.rf, cni.val[,cn.input.attrs]), cni.val$ViolentCrimesPerPop)

varImpPlot(cn.rf, type=1)

cn.attr.utl <- sort(importance(cn.rf)[,1], decreasing=TRUE)
cn.asets <-
  `names<-`(lapply(c(10, 25, 50, 100),
                   function(p)
                   names(cn.attr.utl)[1:round(p*length(cn.attr.utl)/100)]),
            paste("as", c(10, 25, 50, 100), "p", sep=""))

cn.attr.cor <- sort(abs(cor(cn.train[,cn.input.attrs], cn.train$ViolentCrimesPerPop,
                         method="spearman", use="pairwise.complete.obs")[,1]),
                 decreasing=TRUE)
cn.asets <- c(cn.asets,
              `names<-`(lapply(c(10, 25, 50, 100),
                     function(p)
                     names(cn.attr.cor)[1:round(p*length(cn.attr.cor)/100)]),
              paste("as", c(10, 25, 50, 100), "p.cor", sep="")))

cn.tree.as <-
  lapply(cn.asets,
         function(as)
         {
           tree.d <- rpart(make.formula("ViolentCrimesPerPop", as), cn.train)
           tree.f <- rpart(make.formula("ViolentCrimesPerPop", as), cn.train,
                           minsplit=2, cp=0)
           tree.pmin <- prune(tree.f, cpmin(tree.f$cptable))
           tree.p1sd <- prune(tree.f, cp1sd(tree.f$cptable))
           list(tree.d=tree.d,
                r2.d=r2(predict(tree.d, cn.val), cn.val$ViolentCrimesPerPop),
                tree.pmin=tree.pmin,
                r2.pmin=r2(predict(tree.pmin, cn.val), cn.val$ViolentCrimesPerPop),
                tree.p1sd=tree.p1sd,
                r2.p1sd=r2(predict(tree.p1sd, cn.val), cn.val$ViolentCrimesPerPop))
         })

sapply(cn.tree.as,
       function(ta) c(r2.d=ta$r2.d, r2.pmin=ta$r2.pmin, r2.p1sd=ta$r2.p1sd))

prp(cn.tree.as$as10p$tree.d, varlen=0, faclen=0)

cn.lm.as <-
  lapply(cn.asets,
         function(as)
         {
           lmod <- lm(make.formula("ViolentCrimesPerPop", as), cni.train)
           list(lm=lmod,
                r2=r2(predict(lmod, cni.val), cni.val$ViolentCrimesPerPop))
         })

sapply(cn.lm.as, function(ta) ta$r2)

cn.mtree <- lmrpart(make.formula("ViolentCrimesPerPop", cn.asets[["as10p"]]),
                    cn.train, cp=0.02, skip.attr=TRUE))

r2(predict(cn.mtree, cni.val), cni.val$ViolentCrimesPerPop)
##############
#20-4.R
##############
library(rpart)
library(rpart.plot)
library(randomForest)
library(cluster)

library(dmr.claseval)
library(dmr.cluseval)
library(dmr.trans)
library(dmr.util)

covtype <- read.table("../Data/covtype.data", sep=",",
                      col.names=c("Elevation",
                                  "Aspect",
                                  "Slope",
                                  "Horizontal.Distance.To.Hydrology",
                                  "Vertical.Distance.To.Hydrology",
                                  "Horizontal.Distance.To.Roadways",
                                  "Hillshade.9am",
                                  "Hillshade.Noon",
                                  "Hillshade.3pm",
                                  "Horizontal.Distance.To.Fire.Points",
                                  paste("Wilderness.Area", 1:4, sep=""),
                                  paste("Soil.Type", 1:40, sep=""),
                                  "Cover.Type"))

ct.input.attrs <- setdiff(names(covtype), "Cover.Type")

covtype$Cover.Type <- as.factor(covtype$Cover.Type)

set.seed(12)

rct <- runif(nrow(covtype))
ct.train <- covtype[rct>=0.5,]
ct.val <- covtype[rct>=0.25 & rct<0.5,]
ct.test <- covtype[rct<0.25,]

table(ct.train$Cover.Type)/nrow(ct.train)

ct.tree.d <- rpart(Cover.Type~., ct.train, xval=0)
ct.tree.d.pred <- predict(ct.tree.d, ct.val, type="c")
err(ct.tree.d.pred, ct.val$Cover.Type)
confmat(ct.tree.d.pred, ct.val$Cover.Type)

ct.tree.d.cm01 <- confmat01(ct.tree.d.pred, ct.val$Cover.Type)
ct.tree.d.tpfp <- sapply(ct.tree.d.cm01,
                         function(cm) c(tpr=tpr(cm), fpr=fpr(cm), fm=f.measure(cm)))

rowMeans(ct.tree.d.tpfp)

apply(ct.tree.d.tpfp, 1, weighted.mean, table(ct.val$Cover.Type))

ct.tree.f <- rpart(Cover.Type~., ct.train, minsplit=2, cp=0)

ct.tree.f.pred <- predict(ct.tree.f, ct.val, type="c")
ct.tree.f.cm01 <- confmat01(ct.tree.f.pred, ct.val$Cover.Type)
ct.tree.f.tpfp <- sapply(ct.tree.f.cm01,
                         function(cm) c(tpr=tpr(cm), fpr=fpr(cm), fm=f.measure(cm)))
rowMeans(ct.tree.f.tpfp)

ct.tree.pmin <- prune(ct.tree.f, cpmin(ct.tree.f$cptable))
ct.tree.pmin.pred <- predict(ct.tree.pmin, ct.val, type="c")
ct.tree.pmin.cm01 <- confmat01(ct.tree.pmin.pred, ct.val$Cover.Type)
ct.tree.pmin.tpfp <- sapply(ct.tree.pmin.cm01,
                            function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                           fm=f.measure(cm)))
rowMeans(ct.tree.pmin.tpfp)

ct.tree.p1sd <- prune(ct.tree.f, cp1sd(ct.tree.f$cptable))
ct.tree.p1sd.pred <- predict(ct.tree.p1sd, ct.val, type="c")
ct.tree.p1sd.cm01 <- confmat01(ct.tree.p1sd.pred, ct.val$Cover.Type)
ct.tree.p1sd.tpfp <- sapply(ct.tree.p1sd.cm01,
                            function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                           fm=f.measure(cm)))
rowMeans(ct.tree.p1sd.tpfp)

c(default=nrow(ct.tree.d$frame), full=nrow(ct.tree.f$frame),
  pruned.min=nrow(ct.tree.pmin$frame), pruned.1sd=nrow(ct.tree.p1sd$frame))

ct.tree.w <- rpart(Cover.Type~., ct.train, xval=0,
                   parms=list(prior=rep(1/nlevels(ct.train$Cover.Type),
                                         nlevels(ct.train$Cover.Type))))

ct.tree.w.pred <- predict(ct.tree.w, ct.val, type="c")
ct.tree.w.cm01 <- confmat01(ct.tree.w.pred, ct.val$Cover.Type)
ct.tree.w.tpfp <- sapply(ct.tree.w.cm01,
                         function(cm) c(tpr=tpr(cm), fpr=fpr(cm), fm=f.measure(cm)))
rowMeans(ct.tree.w.tpfp)

ct.tree.w.f <- rpart(Cover.Type~., ct.train, minsplit=2, cp=0,
                     parms=list(prior=rep(1/nlevels(ct.train$Cover.Type),
                                          nlevels(ct.train$Cover.Type))))

ct.tree.w.f.pred <- predict(ct.tree.w.f, ct.val, type="c")
ct.tree.w.f.cm01 <- confmat01(ct.tree.w.f.pred, ct.val$Cover.Type)
ct.tree.w.f.tpfp <- sapply(ct.tree.w.f.cm01,
                           function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                          fm=f.measure(cm)))
rowMeans(ct.tree.w.f.tpfp)

ct.tree.w.p1sd <- prune(ct.tree.w.f, cp1sd(ct.tree.w.f$cptable))
ct.tree.w.pmin <- prune(ct.tree.w.f, cpmin(ct.tree.w.f$cptable))

ct.tree.w.pmin <- rpart(Cover.Type~., ct.train, xval=0,
                        minsplit=2, cp=cpmin(ct.tree.w.f$cptable),
                        parms=list(prior=rep(1/nlevels(ct.train$Cover.Type),
                                             nlevels(ct.train$Cover.Type))))
ct.tree.w.p1sd <- rpart(Cover.Type~., ct.train, xval=0,
                        minsplit=2, cp=cp1sd(ct.tree.w.f$cptable),
                        parms=list(prior=rep(1/nlevels(ct.train$Cover.Type),
                                             nlevels(ct.train$Cover.Type))))

ct.tree.w.pmin.pred <- predict(ct.tree.w.pmin, ct.val, type="c")
ct.tree.w.pmin.cm01 <- confmat01(ct.tree.w.pmin.pred, ct.val$Cover.Type)
ct.tree.w.pmin.tpfp <- sapply(ct.tree.w.pmin.cm01,
                              function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                             fm=f.measure(cm)))
rowMeans(ct.tree.w.pmin.tpfp)

ct.tree.w.p1sd.pred <- predict(ct.tree.w.p1sd, ct.val, type="c")
ct.tree.w.p1sd.cm01 <- confmat01(ct.tree.w.p1sd.pred, ct.val$Cover.Type)
ct.tree.w.p1sd.tpfp <- sapply(ct.tree.w.p1sd.cm01,
                              function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                             fm=f.measure(cm)))
rowMeans(ct.tree.w.p1sd.tpfp)

c(weighted.pruned.min=nrow(ct.tree.w.pmin$frame),
  weighted.pruned.1sd=nrow(ct.tree.w.p1sd$frame))

rp.1k <- multi.class(rpart, predf=function(...) predict(...)[,2],
                     encode=multi.enc.1ofk, decode=multi.dec.1ofk)

ct.tree.1k <- rp.1k$alg(Cover.Type~., ct.train, xval=0)

ct.tree.1k.pred <- rp.1k$predict(ct.tree.1k, ct.val)
ct.tree.1k.cm01 <- confmat01(ct.tree.1k.pred, ct.val$Cover.Type)
ct.tree.1k.tpfp <- sapply(ct.tree.1k.cm01,
                          function(cm) c(tpr=tpr(cm), fpr=fpr(cm), fm=f.measure(cm)))
rowMeans(ct.tree.1k.tpfp)

sapply(ct.tree.1k$binmodels, function(m) nrow(m$frame))

ct.tree.1k.cp <- rp.1k$alg(Cover.Type~., ct.train, xval=0, minsplit=2, cp=1e-5)

ct.tree.1k.cp.pred <- rp.1k$predict(ct.tree.1k.cp, ct.val)
ct.tree.1k.cp.cm01 <- confmat01(ct.tree.1k.cp.pred, ct.val$Cover.Type)
ct.tree.1k.cp.tpfp <- sapply(ct.tree.1k.cp.cm01,
                             function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                            fm=f.measure(cm)))
rowMeans(ct.tree.1k.cp.tpfp)

rp.1k.pmin <- multi.class(rpart.pmin, predf=function(...) predict(...)[,2],
                          encode=multi.enc.1ofk, decode=multi.dec.1ofk)
ct.tree.1k.pmin <- rp.1k.pmin$alg(Cover.Type~., ct.train)

ct.tree.1k.pmin.pred <- rp.1k$predict(ct.tree.1k.pmin, ct.val)
ct.tree.1k.pmin.cm01 <- confmat01(ct.tree.1k.pmin.pred, ct.val$Cover.Type)
ct.tree.1k.pmin.tpfp <- sapply(ct.tree.1k.pmin.cm01,
                               function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                              fm=f.measure(cm)))
rowMeans(ct.tree.1k.pmin.tpfp)

ct.tree.pmin.test.pred <- predict(ct.tree.pmin, ct.test, type="c")
ct.tree.pmin.test.cm01 <- confmat01(ct.tree.pmin.test.pred, ct.test$Cover.Type)
ct.tree.pmin.test.tpfp <- sapply(ct.tree.pmin.test.cm01,
                                 function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                                fm=f.measure(cm)))
rowMeans(ct.tree.pmin.test.tpfp)

ct.tree.w.pmin.test.pred <- predict(ct.tree.w.pmin, ct.test, type="c")
ct.tree.w.pmin.test.cm01 <- confmat01(ct.tree.w.pmin.test.pred, ct.test$Cover.Type)
ct.tree.w.pmin.test.tpfp <- sapply(ct.tree.w.pmin.test.cm01,
                                   function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                                  fm=f.measure(cm)))
rowMeans(ct.tree.w.pmin.test.tpfp)

ct.nrmm <- nrm.all(Cover.Type~., ct.train)
ctn.train <- predict.nrm(ct.nrmm, ct.train)
ctn.val <- predict.nrm(ct.nrmm, ct.val)
ctn.test <- predict.nrm(ct.nrmm, ct.test)

ctn.cla <-
  `names<-`(lapply(2:10, function(k)
                         clara(ctn.train[,ct.input.attrs], k,
                               samples=100, sampsize=200, keep.data=FALSE)), 2:10)

plot(2:10, sapply(ctn.cla, function(cm) cm$silinfo$avg.width),
     type="l", xlab="k", ylim=c(0, 0.5))
lines(2:10, sapply(ctn.cla, function(cm) sd(cm$silinfo$clus.avg.widths)), lty=2)
legend("bottomright", legend=c("average silhouette width",
                               "sd(cluster silhouette widths)"), lty=1:2)

par(mfrow=c(1, 3))
plot(silhouette(ctn.cla[["2"]]), main="k=2")
plot(silhouette(ctn.cla[["7"]]), main="k=7")
plot(silhouette(ctn.cla[["10"]]), main="k=10")

ctn.cla7.pred <- predict(ctn.cla[["7"]], ctn.val[,ct.input.attrs])

ct.cla7.tree <- rpart(make.formula("as.factor(ctn.cla[[\"7\"]]$clustering)",
                                    ct.input.attrs),
                       ct.train, xval=0)

ct.cla7.tree.pred <- predict(ct.cla7.tree, ct.val, type="c")
ct.cla7.tree.cm01 <- confmat01(ct.cla7.tree.pred, as.factor(ctn.cla7.pred))
ct.cla7.tree.tpfp <- sapply(ct.cla7.tree.cm01,
                            function(cm) c(tpr=tpr(cm), fpr=fpr(cm),
                                           fm=f.measure(cm)))
rowMeans(ct.cla7.tree.tpfp)

prp(ct.cla7.tree, varlen=0, faclen=0)
