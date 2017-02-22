#http://www.r-bloggers.com/linear-regression-by-gradient-descent/
##
## Linear regression based on the gradient descent
## https://www.coursera.org/learn/machine-learning/lecture/kCvQc/gradient-descent-for-linear-regression
## http://cs229.stanford.edu/notes/cs229-notes1.pdf
## http://machinelearningmastery.com/convex-optimization-in-r/


################################################################################
# GRADIENT DESCENT - in 1D derivative - for function 1D

myfunction <- function(x) {                 # define the function 1D with global optimum (0,0)
  x^2
}

myderivative <- function(x) {               # define your function derivative 1D
  2*x
}

# gradient descent in 1D
gradient_descent <- function(func, myderivative, start, alpha=0.05, tol=1e-8) {
  pt1 <- start                              # the start pt1 1D value
  grdnt <- myderivative(pt1)                # my derivative 
  pt2 <- pt1 - alpha*grdnt                  # substract from the start position the rate alpha times the derivative in this point
  											# this means it goes in the direction of the steepest descent with as small steps as the rate
  while (abs(func(pt1)-func(pt2)) > tol) {  # if subsequent points are very close to each other, 
    pt1 <- pt2                              # then they are near the optimum point and the execution of the loop is stopped
    grdnt <- myderivative(pt1)              # the point 2 becomes the point 1 and again the same process
    pt2 <- pt1 - alpha*grdnt                # if derivative is negative, the point value increases, positive - decreases
    print(func(pt2))                        # print the last in the current loop point my function value
  }                                          
  pt2                                       # return the last point after the loop break
}

result <- gradient_descent(                 # find minimum with the gradient steepest descent
  myfunction,                               # my function to optimize
  myderivative,                             # my function gradient 
  c(runif(1,-3,3)),                         # my starting point
  0.05,                                     # my optimization rate (alpha)
  1e-8)                                     # my admissible error - the distance between subsequent point function values 

# displays optimization outcome
print(result)                               # minimum coordinates
print(myfunction(result))                   # the function value in the optimum
x <- seq(-3, 3, length.out=100)				# generate the x domain
y <- sapply(x, myfunction)                  # just: y <- myfunction(x)
plot(x, y, xlab="x",ylab="y")               # plot my function 
# the optimum marked by the red point 
points(result, myfunction(result), col="red", pch=19)
Sys.sleep(2)                                # 2 second pause


################################################################################
# GRADIENT DESCENT in 2D

myfunction <- function(x) {                 # define the function 2D with global optimum (0,0)
  x[1]^2 + x[2]^2
}

gradient <- function(x) {                   # define your function derivative 2D
  c(2*x[1], 2*x[2])
}


# gradient descent in 2D
gradient_descent <- function(func, gradient, start, alpha=0.05, tol=1e-8) {
  optimhistory <- list()
  pt1 <- start                              # starting point coordinates
  grdnt <- gradient(pt1)                    # my gradient descent in x and y dimensions
  pt2 <- c(pt1[1] - alpha*grdnt[1], pt1[2] - alpha*grdnt[2])
  i <- 1
  optimhistory[[i]] <- pt1
  while (abs(func(pt1)-func(pt2)) > tol) {  # if subsequent points are very close to each other,
    pt1 <- pt2                              # then stop the loop execution at the optimum point.
    grdnt <- gradient(pt1)                  # if derivative in one dimension is negative, 
    pt2 <- c(pt1[1] - alpha*grdnt[1], pt1[2] - alpha*grdnt[2]) #the point value in this dimension increases, positive - decreases
    i <- i + 1;
    optimhistory[[i]] <- pt2
  }
  list(optimhistory,pt2)                    # return the last point after the loop break
}

gradient_result <- gradient_descent(        # find minimum with the gradient steepest descent
  myfunction,                               # my function to optimize
  gradient,                                 # my function gradient
  c(runif(1,-3,3), runif(1,-3,3)),          # my starting point
  0.05,                                     # my optimization rate (alpha)
  1e-8)                                     # my admissible error - the distance between subsequent point function values 

result <- gradient_result[[2]]
optimhistory <- gradient_result[[1]]
# displays optimization results
print(result)                               # optimum coordinates
print(myfunction(result))                   # the function value in the optimum
x <- seq(-3, 3, length.out=100)
y <- seq(-3, 3, length.out=100)
z <- myfunction(expand.grid(x, y))          # print the 2d function
contour(x, y, matrix(z, length(x)), xlab="x",ylab="y")
# the optimum marked by the red point in the middle of the black square
points(result[1], result[2], col="red", pch=19)
#the optimization path to the optimum point
points(optimhistory[[1]][1], optimhistory[[1]][2], col="red", pch=19)
points(optimhistory[[2]][1], optimhistory[[2]][2], col="red", pch=19)
points(optimhistory[[3]][1], optimhistory[[3]][2], col="red", pch=19)
points(optimhistory[[5]][1], optimhistory[[5]][2], col="red", pch=19)
points(optimhistory[[10]][1], optimhistory[[10]][2], col="red", pch=19)
points(optimhistory[[15]][1], optimhistory[[15]][2], col="red", pch=19)
points(optimhistory[[30]][1], optimhistory[[30]][2], col="red", pch=19)
# frame this point with a square
rect(result[1]-0.2, result[2]-0.2, result[1]+0.2, result[2]+0.2, lwd=2)
Sys.sleep(2)                                # 2 second pause



################################################################################
# LINEAR REGRESSION
# generate random data, where y is a random function of the random input x variable (different distributions) 
x <- runif(1000, -5, 5)
y <- x + rnorm(1000) + 3

res <- lm( y ~ x )                          # generate linear regression model 

# a plot of data and the regression line
plot(x,y, col=rgb(0.2,0.4,0.6,0.4), main='Linear Regression')
abline(res, col='blue')
Sys.sleep(2)                                # 2 second pause


################################################################################
# The cost function - the error (between y and the regression line: y - (Ax+B)) square sum 
cost <- function(X, y, theta) {
  sum( (X %*% theta - y)^2 ) / (2*length(y))
}

alpha <- 0.01                               # rate of learning - alpha
num_iters <- 1000                           # an iteration number
cost_history <- double(num_iters)           # a history of cost
theta_history <- list(num_iters)
theta <- matrix(c(0,0), nrow=2)             # initialize regression coefficients - theta
X <- cbind(1, matrix(x))                    # add ones for coefficient B from the equation Ax+B 

for (i in 1:num_iters) {                    # gradient descent for the y distance square sum 
  error <- (X %*% theta - y)                # between points (x,y) and the regression line A*X+B 
  delta <- t(X) %*% error / length(y)       # where Theta is just a vector made of regression coefficients [B,A]
  theta <- theta - alpha * delta            # increasing or decreasing theta depending on the gradient value
  cost_history[i] <- cost(X, y, theta)      # influences the cost function value
  theta_history[[i]] <- theta               # a history of theta values
}

# the data plot and optimization trial
plot(x,y, col=rgb(0.2,0.4,0.6,0.4), main='Linear regression by gradient descent')
for (i in c(1,3,6,10,14,seq(20,num_iters,by=10))) {
  abline(coef=theta_history[[i]], col=rgb(0.8,0,0,0.3))
}
abline(coef=theta, col="blue")
Sys.sleep(2)                                # 2 second pause

cost_history[seq(1,num_iters, by=100)]      # the cost function plot 
plot(cost_history, type='l', col='blue', lwd=2, main='Funkcja kosztu', ylab='koszt', xlab='Iteracja')
Sys.sleep(2)                                # 2 second pause


