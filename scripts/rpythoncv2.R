source('imageFunctions.R')
if (!require(devtools)) {
  install.packages("devtools")
}
if (!require(videoplayR)) {
devtools::install_github("swarm-lab/videoplayR")}
if (!require(imager)) {
devtools::install_github("dahtah/imager")}
if (!require(reticulate)) {
devtools::install_github("rstudio/reticulate")}

library(videoplayR)
imgr = readImg("twosmiles.png")
writeImg("originalWebcamShot.png",imgr)
# Take a picture and save it
#imgr = webcamImage(rollFrames = 10, 
#                   showImage = FALSE,
#                   saveImageToWD = 'originalWebcamShot.png')

# Run Python script to detect faces, draw rectangles, return new image
system('python3 facialRecognition.py')

# Read in new image
img.face = readImg("modifiedWebcamShot.png")

# Display images
imshow(imgr)
imshow(img.face)
str(img.face)

library(png)
pp <- readPNG("modifiedWebcamShot.png")

dev.off()
r <- matrix(runif(9, 0, 1), 3)
str(r)
g <- matrix(runif(9, 0, 1), 3)
b <- matrix(runif(9, 0, 1), 3)
col <- rgb(r, g, b)
dim(col) <- dim(r)
str(col)
library(grid)
grid.raster(col, interpolate=FALSE)
dev.off()
grid.raster(pp, interpolate=FALSE)


library(reticulate)
os <- import("os")
os$chdir(".")
os$getcwd()
cv2 <- import("cv2")
imgp = cv2$imread('originalWebcamShot.png')
str(imgp)
image(t(apply(imgp[,,1],2,rev)), axes=FALSE, col = grey(seq(0, 1, length = 256)))
imgp01<-imgp/256
dev.off()
plot(c(0, 640), c(0, 480), type = "n", xlab = "", ylab = "")
rasterImage(imgp01[,,3],0,0,640,480)


library(abind)
str(imgp01)
str(imgp01[,,1])
imgconv<-abind(imgp01[,,3],imgp01[,,2],imgp01[,,1], along=3)
str(imgconv)
dev.off()
grid.raster(imgconv, interpolate=FALSE)

# https://tutel.me/c/programming/questions/44379525/r+reticulate+how+do+i+clear+a+python+object+from+memory
# https://rstudio.github.io/reticu…/articles/introduction.html
# import numpy and specify no automatic Python to R conversion
np <- import("numpy", convert = FALSE)
# do some array manipulations with NumPy
a <- np$array(c(1:4))
sum <- a$cumsum()
# convert to R explicitly at the end
py_to_r(sum)
