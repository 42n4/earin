#########################################################################################################################
#R neural networks - basic network with backpropagation, training XOR function
#https://en.wikipedia.org/wiki/Backpropagation
#https://aimatters.wordpress.com/2015/12/19/a-simple-neural-network-in-octave-part-1/
#https://yihui.name/knitr/
#library(knitr)
#knit('yourfile.Rnw')
#DEEP LEARNING
#https://www.r-bloggers.com/deep-learning-in-r-2/
#http://www.deeplearningbook.org/contents/mlp.html
#http://deeplearning.net/reading-list/
#REINFORCEMENT LEARNING IN R
#http://www.rblog.uni-freiburg.de/2017/02/26/reinforcement-learning-in-r/
#http://www.rblog.uni-freiburg.de/2017/04/08/reinforcementlearning-a-package-for-replicating-human-behavior-in-r/
#Included samples will run after installing needed packages R 
#(in linux rather on root in the R console to avoid installing locally):


pkglist<-c("clusterGeneration","corrplot","nnet","neuralnet","RSNNS","reshape","rockchalk","fifer","ade4","sqldf","plyr","dplyr")
pkgcheck <- pkglist %in% row.names(installed.packages())
pkglist[!pkgcheck]
for(i in pkglist[!pkgcheck]){install.packages(i,depend=TRUE)}

dev.off()
par(mfrow = c(3, 1)) #three pictures in one column


#set the beginning of the random generator in order to obtain the same results at each run
seed.val <- 1234
set.seed(seed.val)

#the sigmoid threshold function
sigmoid <- function(x) {
  1.0 / (1.0 + exp(-x))
}
#generates x from -10 to 10 step 0.1
x <- seq(-10,10,0.1)
par(mar = rep(2, 4))
plot(x,sigmoid(x))

#input of the exemplary neural network BIAS=1, X1=0, X2=0, where bias is equal to 1, but its weight is changing
A1=c(1,0,0)
#the given output y given the input X=(0,0) - the training pair for the network
y <- 0
#two ways of 2D matrix generation
#in this case the matrix of all connections between the input vector <X1, X2, b> and 3 hidden layer neurons
ncols=3
mrows=3 
W1<-t(replicate(mrows, runif(ncols,-1,1)))
W1<-matrix(runif(mrows*ncols), ncol=ncols)

#in this case the matrix of all connections between the hidden layer (the vector A2 plus b2) and the output neuron 
ncols2=4
mrows2=1 
W2<-matrix(runif(mrows2*ncols2), ncol=ncols2)

#THE FIRST LOOP - FORWARD PROPAGATION OF A
N2 <- W1 %*% A1
A2 <- c(1, sigmoid(N2))
N3 <- c(W2 %*% A2)
h <- A3 <- sigmoid(N3)
h

#THE NEXT LOOP FOR BACKPROPAGATION IN ORDER TO change weights and minimize the error MSE
alfa<-10
J <- ((y * log(h)) + ((1 - y) * log(1 - h))) * -1
delta3 = (h - y)*h*(1-h)
#derivative of sigmoid(Z) is equal to sigmoid(Z)*(1-sigmoid(Z))
#two ways of computing the next layer error derivative
#this one does not work in functions
#delta2<-(delta3 * t(W2) * A2 * (1 - A2))[-1]
#that one is more robust
delta2<-(t(W2) %*% delta3 * A2 * (1 - A2))[-1]

W2<-W2-alfa*delta3%*%t(A2)
W1<-W1-alfa*delta2%*%t(A1)
#FORWARD PROPAGATION OF A
N2 <- W1 %*% A1
A2 <- c(1, sigmoid(N2))
N3 <- c(W2 %*% A2)
h <- A3 <- sigmoid(N3)
h
#END OF LOOP

#set the beginning of the random generator in order to obtain the same results at each run
seed.val <- 1234567890
set.seed(seed.val)
#########################################################################################################################
# APPROXIMATION OF THE XOR FUNCTION ##############################################################################
#########################################################################################################################
# neural network training of the XOR function
xor_nn <-
  function(XOR,
           W1,
           W2,
           init_w = 0,
           learn  = 0,
           alpha  = 0.01) {
    # check whether weights needs initialization
    if (init_w == 1) {
      W1 <- matrix(runif(mrows * ncols), ncol = ncols)
      W2 <- matrix(runif(mrows2 * ncols2), ncol = ncols2)
    }
    # weight corrections from two layers: hidden and output ones
    T1_DELTA = array(0L, dim(W1))
    T2_DELTA = array(0L, dim(W2))
    # go through the whole training set
    m <- 0
    # cost function
    J <- 0.0
    #learned xor
    result<-c()
    #disp('NN output ');
    for (i in 1:nrow(XOR)) {
      # signal forward propagation for output i=1
      A1 = c(1, XOR[i, 1:2])
      N2 <- W1 %*% A1
      A2 <- c(1, sigmoid(N2))
      N3 <- W2 %*% A2
      h <- sigmoid(N3)
      J <- J + (XOR[i, 3] * log(h)) + ((1 - XOR[i, 3]) * log(1 - h))
      m <- m + 1
      
      # computing corrections t2_delta and t1_delta, in order to make error smaller
      if (learn == 1) {
        delta3 = (h - XOR[i, 3])*h*(1-h)
        #derivative of sigmoid(Z) is equal to sigmoid(Z)*(1-sigmoid(Z))
        delta2 <- (t(W2) %*% delta3 * A2 * (1 - A2))[-1]
        # add corrections for all trained input pairs: input - output.
        T2_DELTA <- T2_DELTA + delta3 %*% t(A2)
        T1_DELTA <- T1_DELTA + delta2 %*% t(A1)
      }
      else{
        cat('Hypothesis XOR for ', XOR[i, 1:2], 'equals ', h, '\n')
      }
      result<-c(result,h)
    }
    J <- J / -m
    #cat('delta3: ', delta3, '\n')
    if (learn == 1) {
      W2 <- W2 - alfa * (T2_DELTA / m)
      W1 <- W1 - alfa * (T1_DELTA / m)
      #cat(W2,'\n');
      #cat(W1,'\n');
    }
    else{
      cat('J: ', J, '\n')
    }
    list(W1,W2,result)
  }

#XOR function table for training: two first parameters are inputs, the third one is the output
XOR <- rbind(c(0, 0, 0), c(0, 1, 1), c(1, 0, 1), c(1, 1, 0))

#http://stackoverflow.com/questions/1826519/function-returning-more-than-one-value
#improves collections of object lists from functions
list <- structure(NA, class = "result")
"[<-.result" <- function(x, ..., value) {
  args <- as.list(match.call())
  args <- args[-c(1:2, length(args))]
  length(value) <- length(args)
  for (i in seq(along = args)) {
    a <- args[[i]]
    if (!missing(a))
      eval.parent(substitute(a <- v, list(a = a, v = value[[i]])))
  }
  x
}

#execute with initialization and training
list[W1, W2,] <- xor_nn(XOR, W1, W2, 1, 1, 0.05)

for (i in 1:50000) {
  #execute with training and without initialization
  list[W1, W2,] <- xor_nn(XOR, W1, W2, 0, 1, 0.05)
  if (i %% 1000 == 0) {
    cat('Iteration : ', i, '\n')
    #execute without initialization and training, just an answer of the trained neural network
    list[W1, W2,nauczone_xor] <- xor_nn(XOR, W1, W2)
  }
}
#results should be the same
for (i in 1:nrow(XOR)) {
  cat('The XOR output for ', XOR[i, 1:2], 'equals ', XOR[i, 3], '\n')
}
(XOR[, 3] - nauczone_xor) ^ 2                #square differences between the training set output and the net result 
sum((XOR[, 3] - nauczone_xor) ^ 2)           #Sum Squared Error
#the final net error  
rootofsse <- sqrt(sum((XOR[, 3] - nauczone_xor) ^ 2))
cat('The XOR function training net error', rootofsse,'\n')
Sys.sleep(2)                                 # pause for 2 seconds


#########################################################################################################################
#R neural networks from a package nnet - training XOR function inputs and outputs 
library(clusterGeneration)
library(corrplot)
#import net virtualization functions from github
library(devtools)
source_url(
  'https://gist.github.com/fawda123/7471137/raw/cd6e6a0b0bdb4e065c597e52165e5ac887f5fe95/nnet_plot_update.r'
)
#library nnet
library(nnet)

rand.vars <- data.frame(XOR[, 1:2])
names(rand.vars) <- c('X1','X2')
resp <- data.frame(XOR[, 3])
names(resp) <- c('Y1')
dat.in <- data.frame(resp, rand.vars)
dat.in

#the neural network training with 3 neurons (the minimal number - 3 neurons) 
#and linear output sum 
mod1 <- nnet(rand.vars,
             resp,
             data = dat.in,
             size = 3,
             linout = T)
mod1
#show the net, gray are weights less than 0 and black are positive weights
plot.nnet(mod1)
#our net predicts function XOR values
nauczone_xor <- predict(mod1, cbind(XOR[, 1:2]))
nauczone_xor
#it should be values like that
cbind(XOR[, 3])                              
(cbind(XOR[, 3]) - nauczone_xor) ^ 2         #squared errors of prediction
sum((cbind(XOR[, 3]) - nauczone_xor) ^ 2)    #Sum squared roots
#the final net error  
rootofsse <- sqrt(sum((cbind(XOR[, 3]) - nauczone_xor) ^ 2))
cat('The XOR function training nnet error', rootofsse,'\n')
Sys.sleep(7)                                 # pause for 7 seconds


#########################################################################################################################
# APPROXIMATION OF SINUS IN 20 POINTS      ##############################################################################
#########################################################################################################################
#R neural networks from a package nnet - training sinus function inputs and outputs in 20 points 
#function will approximate the rest of the point domain
library(clusterGeneration)
library(corrplot)
#import net virtualization functions from github
library(devtools)
source_url(
  'https://gist.github.com/fawda123/7471137/raw/cd6e6a0b0bdb4e065c597e52165e5ac887f5fe95/nnet_plot_update.r'
)
#library nnet
library(nnet)

#set the beginning of the random generator in order to obtain the same results at each run
seed.val <- 86644
set.seed(seed.val)

#the number of points to train
num.obs <- 20
#numbers of neurons in the first (and the last) layer
max.neurons <- 200

#20 points to train
x1 <- seq(1, num.obs, 1)
#more dense probe for approximation checking
xx1 <- seq(1, num.obs, 0.3)

#data to train, at the picture red dots
y1 <- sin(x1)
#the real function of sinus, at the final picture the green line
yy1 <- sin(xx1)
plot(x1, y1, col = "red")
lines(xx1, yy1, col = "green")

#pack data into dataframes for nnet library: X1 - input,  Y1 - output to train
rand.vars <- data.frame(x1)
names(rand.vars) <- c('X1')
resp <- data.frame(y1)
names(resp) <- c('Y1')
dat.in <- data.frame(resp, rand.vars)
dat.in

#set the beginning of the random generator in order to obtain the same results at each run
set.seed(seed.val)
#training of the neural network with the 20 neuron layer and linear output
mod1 <- nnet(rand.vars,
             resp,
             data = dat.in,
             size = 20,
             linout = T)

par(mfrow = c(3, 1))
#show the net, gray are weights less than 0 and black are positive weights
plot.nnet(mod1)


#checking the computed network on the more dense probe xx1
x1
xx1
ypred <- predict(mod1, cbind(xx1))
plot(xx1, ypred)
#square differences between the training set output yy1 and the net result ypred
squarederror <- (yy1 - ypred) ^ 2
#sum squared error
sumsquarederror <- sum((yy1 - ypred) ^ 2)
#the final net error  
error1 <- sqrt(sumsquarederror)
error1
Sys.sleep(2)                                 # pause for 2 seconds


#########################################################################################################################
#R neural networks from library nnet - influence of hidden layer neuron numbers
errorlist <- list()                          #empty list
#training networks from 4 to max.neurons e.g. 100 neurons in the only layer
for (i in 4:max.neurons) {
  #set the beginning of the random generator in order to obtain the same results at each run
  set.seed(seed.val)
  #training of the neural network with the i (from the loop for) neuron layer and linear output
  mod1 <- nnet(
    rand.vars,
    resp,
    data = dat.in,
    size = i,
    linout = T,
    trace = FALSE
  )
  #checking the computed network on the more dense probe xx1
  ypred <- predict(mod1, cbind(xx1))
  #the final error from rooted sum of squared errors
  error <- sqrt(sum((yy1 - ypred) ^ 2))
  # and adding to the list which indeks+3 means the neuron number
  errorlist <- c(errorlist, error)
}
#converting list to vector
errorvector <- rapply(errorlist, c)
#plotting of the error vector - there is no rising or other trends 
plot(errorvector)
#minimal error
minerror <- min(errorvector)
minerror
#optimise<-which(errorvector %in% c(min(errorvector)))
#and its indeks which started with a position no 1 for 4 neurons
optimsize <- match(min(errorvector), errorvector)
optimsize
Sys.sleep(2)                                 # pause for 2 seconds

#########################################################################################################################
#nnet with the optimal neuron number
#set the beginning of the random generator in order to obtain the same results at each run
set.seed(seed.val)
#again training with the optimal neuron number for the smallest error
mod1 <-
  nnet(
    rand.vars,
    resp,
    data = dat.in,
    size = optimsize + 3,
    linout = T,
    trace = FALSE
  )
#checking the computed network on the more dense probe xx1
ypred <- predict(mod1, cbind(xx1)) #attention! choose xx1 , not x1
error2 <- sqrt(sum((yy1 - ypred) ^ 2))
#it should be the same minimal error as for the previously calculated in the variable minerror 
error2

#the final picture
#red points - original input data 
#yy1 - the given function on the more dense probe xx1, in the final picture the green line
#the black line is the final net approximation for the more dense probe xx1
#
par(mfrow = c(3, 1))
#plot each model
plot.nnet(mod1)
plot(x1, y1, col = "red")       # red points of training input data points
lines(xx1, yy1, col = "green")  # green line as a real data from which red points were taken 
lines(xx1, ypred)               # black line the approximated line as the trained network output
plot(errorvector)

