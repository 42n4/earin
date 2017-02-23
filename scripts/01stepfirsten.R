#########################################################################################################################
# The R minicourse in one shot (R basics, data structures, clustering, classification):
#########################################################################################################################
# Made and assembled from various sources by Piotr Wąsiewicz 
# General rules:
# - indices in R begin with 1, not with 0
# - R language is case sensitive
# - in the rstudio the mouse cursor focus on the function name or F1 show appropriate help
# - all examples in this script need to be executed one after another, later ones may use the previous ones outcomes
# - In windows R is automatically installed, the first step is to install R for windows server
# https://mran.microsoft.com/install/mro4mrs/8.0.0/MRO-3.2.2-for-RRE-8.0.0-Windows.exe
# downloaded from the site https://mran.microsoft.com/download/mro-for-mrs/
# and the rstudio: # https://download1.rstudio.org/RStudio-0.99.902.exe
# taken from the site https://www.rstudio.com/products/rstudio/download/
# - it is necessary to install these packages to run all examples
# - in linux you should install it on the root account to have it available without problems
pkglist<-c("reshape","ade4","sqldf","plyr","dplyr")
pkglist<-c(pkglist,"party","rgl","scatterplot3d","fpc","pvclust","dendextend")
pkglist<-c(pkglist,"nFactors","FactoMineR","RRF","mclust","foreach","doParallel")
pkglist<-c(pkglist,"rpart","ipred","gbm","mda","klaR","kernlab","caret")
pkglist<-c(pkglist,"tseries","fUnitRoots","forecast","sets","TTR")
#pkglist<-c(pkglist,"MASS","RWeka")
pkgcheck <- pkglist %in% row.names(installed.packages())
pkglist[!pkgcheck]
#COMMMENT the line below if you installed packages earlier e.g on root
for(i in pkglist[!pkgcheck]){install.packages(i,depend=TRUE)}
#loading all libraries - necessary to avoid errors of execution
for(i in pkglist) library(i, character.only = TRUE);
#########################################################################################################################
# Basic commands in R
# help function in R
#help(kmeans)                       # shows the kmeans function help
#?kmeans                            # shows the kmeans function help 
#help(pi)                           # displays help of the special pi number
getwd()                             # gives us the working directory name
setwd()                             # sets the working directory to the given in the first argument
dir.create(foldername)              # make a directory 'foldername'
source('file.R')                    # execute the R script 'file.R'
source('file.R', echo = TRUE)       # execute the R script 'file.R' with echo, more verbose information
a <- 2; print(a)                    # print() writes the variable a to the console 
i <- 10                             # with cat you can also write to the console
cat(i, "ta zmienna...\n", sep = "") # sep="" means no whitespaces between joint fragments
#conversion of strings and numerical variables
assign('test', 10)                  # the same execution result as in the command: test <- 10 (or: test = 10)
x <- 'test'                         # assign the string to the variable
assign(x, 5)                        # the same result is after: test <- 5
x <- 5                              # assign the variable 'x' name to 'var.name' with the help of: deparse substitute
var.name <- deparse(substitute(x))  # var.name is equal to "x"
vec <- c(1:10)                      # check the structure of the variable in this case a vector
str(vec)                            # str shows structure of 'vec'
head(vec)                           # returns 6 first elements
tail(vec)                           # gives 6 last elements
no <- c(1:3)
grade <- c(89, 95, 100)
data <- data.frame(no, grade)       # dataframe made of two vectors
head(data)                          # shows 6 first rows
tail(data)                          # outcomes 6 last rows
a <- c(1, 2);class(a)               # 'class' shows the type of the variable 'a'
# conversion types
#is.datatype()                      # returns TRUE or FALSE, whether datatype is e.g. integer i.e. is.integer
#as.datatype()                      # converts to datatype, where datatype is e.g. integer i.e. is.integer
x <- c(1, 2); rm(x)                 # remove variables from working memory, in this case 'x'

# rm(list=ls(all=T))                # remove all variables from working space
rm(list = ls(pattern = '^tmp'))     # remove all libraries beginning with 'tmp'
# quit()                            # exit the rstudio (asks whether to save the actual workspace)


#########################################################################################################################
# Data structures
#########################################################################################################################
# VECTORS has only one datatype within
# indices in R begins with 1, not with 0
a <- c(1, 2, 5, 3, 6,-2, 4)
a[3]                   # 5
a[c(1, 3, 5)]          # 1 5 6
a[2:6]                 # 2 5 3 6 -2
b <- replicate(10, 2)  # generates 'a' vector with the length equal to 10 and every element equal to 2
b <- rep(2, 10)        # generates 'a' vector with the length equal to 10 and every element equal to 2 
b <- 1:10              # 'b' equals 1, 2, 3, 4, 5, 6, 7, 8, 9 10
b <- seq(1, 10)        # 'b' equals 1, 2, 3, 4, 5, 6, 7, 8, 9 10
b <- seq(1, 10, 2)     # 'b' equals 1, 3, 5, 7, 9
b <- seq(0, 1, 0.1)    # 'b' equals 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
b <- seq(from=0, to=1, by=0.1) # 'b' equals 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
e <- 10
a <- c(a, e)           # add the element 'e' to the vector 'a': the method 1
a <- append(a, e)      # add the element 'e' to the vector 'a': the method 2  
a <- append(a, e, 2)   # add 'e' at the position 2 + 1 i.e. 3
a <- append(a, e, 0)   # add 'e' at the position 0 + 1 i.e. 1
a[length(a) + 1] <- e  # add the element 'e' to the vector 'a': the method 3
index <- 2             # index is equal to 2
a <- a[-index]         # remove the vector element at the position index
a[-1]                  # shows vector without the first element
a[-length(a)]          # shows vector without the last element
a[-c(2, 5)]            # remove the second and fifth element from 'a'
a                      #c( 10,  1,    2,     10,   5,   3,     6,   -2,    4,    10,   10,   10)
a > 3                  # TRUE FALSE FALSE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE      
a[a>3]                 # take elements greater than 3
a[a==10]               # select vector elements equal to 10
a[a>1 & a < length(a)] # vector elements greater than 1 and less than the 'a' length 
s<-c(jeden=1,dwa=2,trzy=3)# vector with elements, which have names
names(s)               #elements names
s[["trzy"]]            #choose the element with the name "trzy"
# find index of the first matching element
# e.g. find 10-th element in the vector vec <- c(1, 10, 2, 10).
# match(10, vec) returns 2, the next index of '10' is not displayed
vec <- c(10, 2:10)
e  <- 10
e2 <- c(10, 5)
match(e, vec)          #gives the first matching place of 'e' in the vector 'vec'
which(vec %in% e)      #returns all matching 'e' indices  in the vector 'vec'
match(e2, vec)         #gives the first matching substring 'e2' element index in the vector 'vec'
which(vec %in% e2)     #gives all matching indices of the substring 'e2' in the vector 'vec' 
e3 <- 1                #'e3' equals 1
!is.na(match(e3, vec)) #checks whether 'e3' exists within the vector 'vec'
e3 %in% vec            #checks whether 'e3' exists within the vector 'vec'
c1 <- c(1, 2, 3)
c2 <- c(2, 3, 5)
c1[!(c1 %in% c2)]      # 1 - returns elements from the 'c1' vector not existing in 'c2' in this case: 1
setdiff(c1, c2)        # 1 - the second method
c2[!(c2 %in% c1)]      # 5 - returns elements from the 'c2' vector not existing in 'c1' in this case: 5
setdiff(c2, c1)        # 5 - the second method
vec <- c(1, 2, 3, 2)   # 
nlevels(factor(vec))   # counts unique elements in this case: 3
length(unique(vec))    # returns unique elements in this case: 3
# operators making calculations on elements with the same index
c(1,3,5) + c(5,3,1)    #-> 6,6,6
c(1,3,5) - c(5,3,1)    #-> -4,0,4
c(1,3,5) * c(5,3,1)    #-> 5,9,5
c(2)     * c(5,3,1)    #-> 10,6,2
c(1,3,5) / c(5,3,1)    #-> 0.2,1.0,5.0
c(1,3,5)%/%c(5,3,1)    #-> 0,1,5  integer division - the fractional part is discarded 
c(1,3,5) %%c(5,3,1)    #-> 1,0,0  modulo division - returns the division remainder (not fractional)
c(1,3,5) ^ c(5,3,1)    #-> 1,27,5 the powers in the second vector of the elements in the first one
c(1,3,5) **c(5,3,1)    #-> 1,27,5 the second power method
c(1,3,5)%in%c(5,3,1)   #-> TRUE,TRUE,TRUE checks whether first vector elements exist in the second vector
# vector calculations - sum of element * element 
c(1,3,5) %*% c(5,3,1)  #-> 1*5+3*3+5*1 = 5+9+5 = 19
sort(a)                #vector elements sorting
sort(a, decreasing = TRUE) 
x <- 'abc'
numc <- nchar(x)       # number of chars in a string
numc
# find a char index in a string
# \" is a single character, loc is a list
loc <- gregexpr(pattern = '\"', "abc\"defg") 
cat('Character Position: ', loc[[1]][1], '\n')
# conversion of a string to a number
# 1 method
x <- 123
x <- paste(x)          # x equals "123"
x <- strtoi(x)         # x equals 123
# 2 method
x <- 123
x <- as.character(x)   # x equals "123"
x <- as.integer(x)     # x equals 123, and with the help of  as.numeric(x) as a float
#attention!
c(5,'a'); e<-c(5)      # default conversion c('5','a')
e[2]<-'a'; e           # default conversion c('5','a')
typeof(1:2) == typeof(c(1,2))           # FALSE the first type integer - the second double
for( i in 1:length(c()))        print(i)# 1 0 not so empty vector, it can be printed
for( i in seq_len(length(c()))) print(i)# the correct loop form for empty vectors 
for( i in seq_along(c()))       print(i)# the second correct form


#########################################################################################################################
#MATRIX: two dimensional structure of the same type elements
# use matrix indices
x <- matrix(1:16, nrow = 4)
x
x[2, ]                 #the second row
x[, 3]                 #the third column
x[1, 4]                #an element from the first row and the fourth column
x[1, c(3, 4)]          #elements from the first row and the fourth and fifth columns
nrow(x)                #a row number
ncol(x)                #a column number
dim(x)                 #matrix dimensions c(nrow(x),ncol(x))
length(x)              #nrow(x)*ncol(x)
rowMeans(x)            #averages calculated from rows
colMeans(x)            #averages computed from columns
rowSums(x)             #sums calculated from rows
colSums(x)             #sums computed from columns
t(x)                   #the transpose matrix x
det(x)                 #the determinant of a matrix in this case equal to 0
#creating the column 4x1 from a vector and a matrix 2x2
cells <- c(1, 6, 4, 8)
cells
matrix(cells)          # the column, vertical vector
t(t(cells))            # the transpose two times vector is the column vector
rnames <- c('R1', 'R2')
cnames <- c('C1', 'C2')
colmatrix<-matrix(     # fill the matrix columns with default settings
  cells,
  nrow = 2,
  ncol = 2,
  byrow = FALSE,
  dimnames = list(rnames, cnames)
)
colmatrix
rownames(colmatrix)    #row names
colnames(colmatrix)    #column names
c(colmatrix)           #conversion to an usual vector
rowmatrix<-matrix(     #fill the matrix rows
  cells,
  nrow = 2,
  ncol = 2,
  byrow = TRUE,
  dimnames = list(rnames, cnames)
)
rowmatrix              #it is the same as t(colmatrix) - the transpose matrix
c(rowmatrix)           #conversion to a vector, but another than the original one called cells
# matrix multiplication e.g. the first row and the first column multiplication 
# is just an element by an element multiplications sum 
# which is just an c(1,1) element of the outcome matrix
# and c(1,2) is the first row and the second column multiplication
colmatrix %*% rowmatrix #the multiplication of matrices (row by column): rows of the first one by the columns of the second one
#outcome M is just e.g. the matrix M 2x2 if two input arguments were also 2x2 matrices 
#M[1,1]=colmatrix[1,] %*% rowmatrix[,1] M[1,2]=colmatrix[1,] %*% rowmatrix[,2]
#M[2,1]=colmatrix[2,] %*% rowmatrix[,1] M[2,2]=colmatrix[2,] %*% rowmatrix[,2] 
colmatrix * rowmatrix #matrices multiplication (an element colmatrix([i,j] multiplied by an element rowmatrix[i.j]) 
crossprod(colmatrix,rowmatrix)   #the first transpose matrix (row by column multiplication) t(colmatrix) %*% rowmatrix
tcrossprod(colmatrix,rowmatrix)  #the second transpose matrix (row by column multiplication) colmatrix %*% t(rowmatrix)
solve(colmatrix)                 #inverse of a square matrix
solve(colmatrix,rowmatrix)       #solve the equation colmatrix %*% X = rowmatrix
solve(rowmatrix,colmatrix)       #solve the equation rowmatrix %*% X = colmatrix
solve(rowmatrix,rowmatrix)       #conversion into the same matrix through a diagonal one X=diag(2)
diag(2)                          #the diagonal matrix
det(colmatrix %*% rowmatrix)     #matrices determinants det(A · B) = detA · detB
det(colmatrix) %*% det(rowmatrix)#det(A · B) = detA · detB
#http://www.statmethods.net/advstats/matrix.html

#########################################################################################################################
#ARRAY: similar to the matrix type, but with many dimensions 
dim1 <- c('A1', 'A2')
dim2 <- c('B1', 'B2', 'B3')
dim3 <- c('C1', 'C2', 'C3', 'C4')
z <- array(1:24, c(2, 3, 4), dimnames = list(dim1, dim2, dim3))
z
z[1, 2, 3] #one cell from the 3-dimensional matrix


#########################################################################################################################
#LIST: an ordered set of various type elements
g <- 'Moja pierwsza lista'
h <- c(25, 26, 18, 39)
j <- matrix(1:10, nrow = 5)
k <- c('jeden', 'dwa', 'trzy')
mlist <- list(tytul = g, age = h, j, k)
mlist
mlist[[2]]             #the second list element, in this case: a vector
mlist[['age']]         #the element age, in this case: a vector
mlist$age              #the element age, in this case: a vector
mlist['age']           #the element age, in this case: a list (do not use for a vector)
mlist[2]               #the element age, in this case: a list (do not use for a vector) 
typeof(mlist[['age']]) #a double type 
typeof(mlist['age'])   #a list type
mlist[[2]][[1]]        #25, the first element of the list age 
as.list(h)             #conversion to the list
nlist<-as.list(h)       
list(mlist,nlist)      #the list of two lists
plist<-list(mlist,nlist)
print(plist)           #print the list of lists
str(plist)             #structure of data in this case the list of lists
dput(plist)            #code in R of two layer list
class(plist)           #the class of a type
#cat(plist)            ATTENTION! cat with list does not work!
#two first list elements are mutated
mlist[names(mlist) %in% c('tytul','age')]<-c('Nadpisany element listy',list(c(2,4,6,7)))
lapply(mlist,FUN=length)#use the function 'length' for every list element creating a length LIST
sapply(mlist,FUN=length)#use the function 'length' for every list element creating a length VECTOR
ylist<-list(a=1, b=3, c=5, d=6)
sapply(ylist, FUN=function(x,p) x^p, p=2)   #power outcome for p=2 is the vector 1,9,25,36
sapply(ylist, FUN=function(x,p) x^p, p=2:3) #power outcome for p=2 is the matrix with 2 rows 1,9,25,36 and 1,27,125,216


#########################################################################################################################
#DATAFRAME: named columns with various data types just like an excel sheet or a database table
pacjent_id <- c(1, 2, 3, 4)
age <- c(25, 34, 28, 52)
type <- c('Type1', 'Type2', 'Type1', 'Type1')
state <- c('Poor', 'Progress', 'Excellent', 'Poor')
patient <- data.frame(pacjent_id, age, type, state)
nrow(patient)         # a row number
ncol(patient)         # a column number
dim(patient)          # a dimension vector with row and column numbers c(nrow(patient),ncol(patient))
rownames(patient) <- seq_len(nrow(patient)) #how to name rows with their indices
# how to get a dataframe row or a column
i <- 1; j <- 2
patient[i,]           # the ith row as a vector
patient[, j]          # the jth column as a vector
patient[, 'age']      # a column 'age' as a vector
patient[['age']]      # a column 'age' as a vector  
patient$age           # a column 'age' as a vector
patient[j]            # the jth column as a dataframe
patient['age']        # the column 'age' as a dataframe 
patient[1:2]          # the two first columns as a dataframe
patient[c('typ', 'state')]
# get the ith and jth dataframe cell
as.integer(patient[i,][j])# patient[i, ][j] the ith row and jth column cell as an integer
patient[i, j]         # the ith row and jth column cell 
patient[[i, j]]       # the ith row and jth column cell 
patient[[j]][i]       # the ith row and jth column cell 
patient[, j][i]       # the ith row and jth column cell 
patient$age[i]        # the ith row of the column 'age' 
patient[i, 'age']     # the ith row of the column 'age'
patient[i, i:j]       # two cells from the ith row and i-jth columns, it does not work with double square brackets [[i, i:j]]
# remove a row
index <- 2
patient[-index,]      # remove the second row from the dataframe 'patient'
# choosing subsets
patient[1:3, ]        # three first rows - patient
patient[which(patient$state == 'Poor' & patient$age < 30), ] #patients with the state poor and the age above 30
library(plyr); library(dplyr)               # use package dplyr (install first)
filter(patient, state == 'Poor' & age < 30) # subset()
subset(patient, age >= 35 | age < 24, select = c(age, state))
subset(patient, state == 'Poor' & age < 30, select = pacjent_id:date)
# sorting
patient[order(patient$age), ]               # sort rows from the youngest to the oldest patient, with the ascending order
attach(patient)
spatient <- patient[rev(order(type, age)),] # sort rows in the descending order with 'rev'
detach(patient)
spatient
attach(patient)
spatient <- patient[order(type,-age),]      # sort rows 
detach(patient)
spatient
#Data joining: adding rows
new_row <-                                  # a new row
  data.frame(
    pacjent_id = 5,
    age = 10,
    type = 'Typ3',
    state = 'Zdrowy'
  )
patient <- rbind(patient, new_row)          # RBIND adds a new row to the dataframe
spatient<- rbind(spatient, new_row)         # or to another dataframe
spatient$pacjent_id <- spatient$pacjent_id + 10 #other ids for sorted patients - spatient
rbind(patient, spatient)                    # rbind adds another dataframe to the first one
#Data joining: adding columns
patient$new_col <- c(2:6)                   # add a column to the dataframe: method 1
patient$new_col <- NULL                     # remove a column from the dataframe: method 1
patient <- transform(patient, new_col = c(2:6))# add a column to the dataframe: method 2
patient <- within(patient, {new_col = NULL})# remove a column from the dataframe: method 3
merge(patient, spatient, by = "pacjent_id") # join by the column ID patient and spatient
merge(patient, spatient, 
      by = c('pacjent_id', 'typ'))          # join by columns ID, Country dataframes patient and spatient
cbind(patient, spatient)                    # CBIND join dataframes patient and spatient with the same row number
#Usuwanie kolumn
myvars <- names(spatient) %in% c('age', 'type')# selecting attributes in columns age and type 
myvars
spatient[!myvars]                           # remove columns age, type
spatient$age <- spatient$type <- NULL       # remove columns age, type
spatient[c(-2,-3)]                          # remove 2nd and 3th column
#Adding date
#Sys.Date() returns the object Date, date() returns the text string
patient$date <- Sys.Date()
patient
startdate <- as.Date('2009-01-01')
enddate <- as.Date('2017-10-31')
patient[which(patient$date >= startdate & patient$date <= enddate), ]
subset(patient,date >= startdate & date <= enddate)
#Random sample
sample(1:nrow(patient), 3, replace = FALSE)
patient[sample(1:nrow(patient), 3, replace = FALSE),]
#Adding new attributes to the dataframe
# several methods
df <- data.frame(x1 = c(2, 2, 6, 4), x2 = c(3, 4, 2, 8))
# method 1
df$sumx <-  df$x1 + df$x2
df$meanx <- (df$x1 + df$x2) / 2
# method 2
attach(df)
df$sumx <-  x1 + x2
df$meanx <- (x1 + x2) / 2
df
detach(df)
# method 3
transform(df, sumx = x1 + x2, meanx = (x1 + x2) / 2)
# method 4
with(df, {                                  # 'with' returns nothing
age[x1 == 2] <- 1                           # the rest is filled with NA (lack of data in R)
})
df
df$age <- NULL                              # remove age
# method 5
df <- within(df, {                          # 'within' returns df
  age <- NA                                 # create the new attribute age and initialize it with NA (lack of data in R)
  age[x1 == 2] <- 1
})
df
#Rename
#rename(dataframe, c(oldname1="newname1", oldname2="newname2",...))
#library(reshape)
df <- rename(df, c(sumx = "sum"))
df
names(df)[4] <- "average"                   # names(df) returns the string vector or use fix(df) in gui
#manipulating with NA
df$age[df$age == 1] <- NA                   # changing 1 to NA in the column age
df
df$age[is.na(df$age)] <- 55                 # changing NA to 55 in the column age
df$age <-
  ifelse(is.na(df$age), 55, df$age)         # changing NA to 55 in the column age
df
x <- c(1, 2, NA, 3)                         # a vector filled with NA
y <- x[1] + x[2] + x[3] + x[4]              # y equals NA
z <- sum(x)                                 # y equals NA
z <- sum(x, na.rm = TRUE)                   # na.rm=TRUE remove rows with NA
df$age[df$sum == 6] <- NA
df
na.omit(df)                                 # na.omit() removes rows with NA
df[!is.na(df$age),]                         # removes rows with NA

#########################################################################################################################
#FACTORS - labels, classification concept, discrete variables 
#features e.g. eye colour, sex, blood group
#ordered – e.g. height levels: (low, medium, high)
#discrete – e.g. a number of own children, a farms number, a man age 
#map of discrete vectors [1...k]
#factors cannot be added or multiplied 
#does not work  $, use [] with an index e.g. levels(x)[1]
type <- c('Type1', 'Type2', 'Type1', 'Type1')
type <- factor(typ)                         # Levels: Type1 Type2
typ
state <- c('Poor', 'Progress', 'Excellent', 'Poor')
state <-
  factor(state, ordered = TRUE)             # Excellent-3 Progress-2 Poor-1
state
levels(state)                               #shows levels of state
state2 <-
factor(state,
levels = c('Excellent', 'Progress', 'Poor'))# Excellent-1 Progress-2 Poor-3
state2
levels(state2)                              #shows levels of state2
state2 <- factor(c(as.character(state2), 'Pathetic')) #adds the new level of state
levels(state2)                              #shows levels with a new state level
#Discretization of the variable
i<-1:50+rnorm(50,0,5); i                    #variable i
k<-cut(i,5); k                              #generate k levels from the i
levels(k)<-seq_len(length(levels(k)))       #renames levels
levels(k)
# levels in dataframes - default alphabetic order
patient
str(patient)
summary(patient)
table(patient$type, patient$state)          #generate statistics from two factor columns
#count averages
#library(reshape)
#melt and cast or recast, ddply, aggregate i.e. 4 METHODS of aggregation
library(reshape)
patient$date <- NULL
# for every pair of type and state write in one column aggregations
md <- melt(patient, id = (c('type', 'state'))) 
md
cast(md, state ~ variable, mean)            #calculate state averages 
cast(md, type ~ variable, mean)             #type means
cast(md, type + state ~ variable, mean)     #averages for type and state values
recast(patient, type + state ~ variable, 
       mean, id.var = c('type', 'state'))   #type and state averages
ddply(patient, ~type + state, 
      summarise, N=length(age), 
      sredniaid=mean(pacjent_id),
      sredniaage=mean(age))                 #other variables averages of type and state 
ddply(patient, .(type,state), 
      summarise, N=length(age), 
      sredniaid=mean(pacjent_id),
      sredniaage=mean(age))                 #other variables averages of type and state
aggregate(.~state+type,data=patient,mean)   #other variables averages of type and state


#########################################################################################################################
#FUNCTION PLOT
f <- function(x) {
  x * sin(x)
}
par(mar = rep(2, 4))
plot(f,-20 * pi, 20 * pi)

#two plots in one picture
x <- c(1:5)
y1 <- 2 * x
y2 <- 3 * x
plot(
  x,
  y1,
  type = 'l',
  col = 'red',
  xlab = 'day',
  ylab = 'Net Value'
)
lines(x, y2, type = 'l', col = 'green')
# label lines with legend(), lwd - line width, bty='n' - no graphical frame
legend(
  'topleft',
  legend = c('line 1', 'line 2'),
  col = c('red', 'green'),
  lwd = 1,
  bty = 'n'
)
#draw many plots in one moment
plot(f)                                     # the first plot
dev.new()                                   # the second plot
plot(f(x2plot/20))                          # switching with keyboard arrows
Sys.sleep(2)                                # 2 seconds pause
dev.off()                                   # close the second draw

#########################################################################################################################
# attach, detach
# attach, detach do not work only on the column names without $, use "with"
summary(mtcars$mpg)
plot(mtcars$mpg, mtcars$disp)
plot(mtcars$mpg, mtcars$wt)
attach(mtcars)                              # add a data set to the working path
summary(mpg)
plot(mpg, disp)
plot(mpg, wt)
detach(mtcars)                              # remove a data set from the working path
Sys.sleep(2)                                # 2 seconds path

#use SQL from databases
library(sqldf)
head(mtcars)
sqldf('select * from mtcars where carb=1 order by mpg', row.names = TRUE)
sqldf(
  'select avg(mpg) as avg_mpg, avg(disp) as avg_disp, gear from mtcars where cyl in (4, 6) group by gear'
)


#########################################################################################################################
#WRITING TO FILES
#export data to the csv file
write.table(
  data,
  'data.csv',
  row.names = F,
  col.names = F,
  sep = ','
)
#add data from the csv file
signal <- read.table('data.csv', header = FALSE, sep = ',')
#WRITING PICTURES 
#the first picture pdf file
pdf('plotgraph1.pdf')
x2plot = seq(1:20)
plot(sin(x2plot))
dev.off()
#the second picture file
pdf('plotgraph2.pdf')
plot(cos(x2plot))
dev.off()


#DATE and TIME
#write date as a variable
# x is a date in the text string form
# the date format %Y-%m-%d
mydate <- as.Date('2016-04-28', '%Y-%m-%d')
mydate
class(mydate)
#write data to the formatted string
# x is a date in a string
# the date format %Y-%m-%d
datechar <- format(mydate, format = '%Y-%m-%d')
datechar
class(datechar)
# calculate the date period in seconds, minutes, hours, days or weeks
# dataX and dataY are date variables
interval <- difftime('2016-04-01', '2016-04-28', units = 'weeks')
interval
# write time in a script
t1 <- proc.time()
for (i in 1:1000) {
  cat('.')
}
t2 <- proc.time()
time_elapsed <- (t2 - t1)[[3]]              #date period
time_elapsed
time_elapsed <- as.numeric((t2 - t1)[3])    #date period
time_elapsed

#########################################################################################################################
#LOOPS, FUNCTIONS
#data flow control
#if/else, ifelse, switch
score <- 0.6
if (score > 0.5) {
  outcome2 <- 'passed'
} else {
  outcome2 <- 'not passed'
}
outcome <- ifelse(score > 0.5, 'passed', 'not passed')
#FOR, WHILE, SAPPLY
for (i in 1:10) {                           #for (var in seq) statement
  print('witaj')                            #write ten times 'witaj'
}
sapply(1:10, function(i) {                  #sapply - alternative for for - write 10 times 'witaj'
  print('witaj')                            #write ten times 'witaj'
})
for (i in 1:10) {                           #write numbers from 1 to 10
  print(i)
}
sapply(1:10,function(i){print(i)})          #sapply - alternative for for - write numbers from 1 to 10
i <- 10                                     #while (cond) statement - begins with i=10
while (i >= 0) {                            #write ten times 'witaj'
  print('witaj')
  i <- i - 1
}
for(i in seq_len(nrow(patient)))            #loop even for empty dataframes
  print(patient$age[i])
#FUNCTIONS
#myfuncton <- function(arg1, arg2, ...) {
#  code, R commands
#  return(object)
#  or just enough
#  object
#}
#more advanced function
mystats <- function(x,
                    parametric = TRUE,
                    print = FALSE) {
  if (parametric) {
    center <- mean(x)
    spread <- sd(x)
  } else {
    center <- median(x)
    spread <- mad(x)
  }
  if (print & parametric) {
    cat('Mean=', center, '\n', 'SD= ', spread, '\n')
  } else if (print & !parametric) {
    cat('Median=', center, '\n', 'MAD = ', spread, '\n')
  }
  result <- list(center = center, spread = spread)
  return(result)
}
mystats(patient$age,TRUE,TRUE)
mystats(patient$age,FALSE,TRUE)
#SAPPLY, APPLY  
fn <- function (x) {                        #function with one argument
  ifelse(x > 46 & x < 52, 1, 0)
}
fn(40:60)                                   #a vector as an argument
sapply(40:60, fn)                           #a vector as an argument
fn <- function (x, y) {                     #two arguments function
  ifelse(x > 46 & x < 52 & y < 12, 1, 0)     
}
datagrid <- expand.grid(i = 40:60, j = 0:20)#create all value combinations i, j; first join j=0 with all values of i 
#fn(datagrid$i, datagrid$j)                 #IT DOES NOT WORK!
apply(datagrid, 1, function(z) {            #just like two loop for i and j
  fn(z["i"], z["j"])                        #apply works for matrices, the second argument 1 means for rows, 2 for columns
})
apply(datagrid,1,function(z){fn(z[1], z[2])})#a short version with indices, 1 means rows for function fn
res <- NULL                                 #loop in loop - just like apply with datagrid
for (j in 0:20) {
  for (i in 40:60) {                        #create all value combinations i, j
    res <- c(res, fn(i, j))                 #expand.grid(i = 40:60, j = 0:20); first join j=0 with all values of i 
  }
}
res

#Advanced functions 
#math functions: sqrt(x), floor(x), log(x), exp(x)
#statistical functions: mean(x), median(x), sd(x), var(x), range(x), sum(x), scale(x, center=TRUE, scale=TRUE)
#probabilistic functions:
#[dpqr]distribution_abbrebiation (d=density, p=distribution function, q=quantile function, r=random generation)
#runif()- próba o równomiernym rozkładzie (random generation)
#set.seed(5): set seed to receive always the same outcome
#string functions:
#nchar(x), nzchar(x), substr(x, start, stop), grep(pattern, x, ignore.case=FALSE, fixed=FALSE)
#sub(pattern, replacement, x, ignore.case=FALSE, fixed=FALSE), strsplit(x, split, fixed=FALSE)
#paste(..., sep=""), toupper(x), tolower(x)
#other functions:
#length(x), seq(from, to, by), rep(x, n), cut(x, n), pretty(x, n), cat(.., file='myfile', append=FALSE)


#########################################################################################################################
#MACHINE LEARNING - DATA RETRIEVAL
#########################################################################################################################
#https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R
#########################################################################################################################

set.seed(12459);                            # the same outcome for repeated script execution 
dev.off()                                   # in any case turn off the second picture

#########################################################################################################################
#DATA ANALYSIS
#chosing data for clustering and classification
#first we list all exemplary data
#data()
#then list its structure
#?nazwa_danych
#choosing:
library(RRF)
ndata <- "imports85"
data(list=ndata)
mydata<-get(ndata)
mydata<-mydata[sapply(mydata, is.numeric)]  # get only numerical data
mydata$normalizedLosses<-NULL               # remove not wanted columns
mydata$symbolic<-NULL
mydata <- na.omit(mydata)                   # remove rows with NA
rownames(mydata) <- seq_len(nrow(mydata))   # rename rows
#scale(x) means (x - mean(x)) / sd(x) from different domains (0,10) and (0,10000) to the only one (0,1)
mydata_scaled <- mydata
mydata_scaled[sapply(mydata, is.numeric)] <- scale(mydata_scaled[sapply(mydata, is.numeric)])

mydata$grupa<-NULL
cor(mydata)                                 # variable correlations
round(cor(mydata), 2)                       # checking correlations if variables increase together or decrease together
#image(cor(mydata))
#heatmap with distances between rows
distMatrix <- as.matrix(dist(mydata_scaled))
heatmap(distMatrix)
Sys.sleep(2)                                # 2 seconds pause
#heatmap from correlations
#https://planspacedotorg.wordpress.com/2013/07/24/clustered-r-squared-heat-maps-in-r/
dissimilarity <- 1 - cor(mydata)^2          #a measure of dissimilarity
clustering <- hclust(as.dist(dissimilarity), method="ward.D2")
plot(clustering)                            #clustering
order <- clustering$order
oldpar <- par(no.readonly=TRUE); par(mar=c(0,0,0,0))
image(dissimilarity[order, rev(order)], axes=FALSE)
par(oldpar)
clusterRsquared <- function(dataframe) {    #dissimilarity function
  dissimilarity <- 1 - cor(dataframe)^2
  clustering <- hclust(as.dist(dissimilarity))
  order <- clustering$order
  oldpar <- par(no.readonly=TRUE); par(mar=c(0,0,0,0))
  image(dissimilarity[order, rev(order)], axes=FALSE)
  par(oldpar)
  return(1 - dissimilarity[order, order])
}
round(clusterRsquared(mydata),2)
#round(clusterRsquared(mydata3kol),2)
#Sys.sleep(2)                               # 2 seconds pause


#########################################################################################################################
#CLUSTERIZATION, CLUSTERING, GROUPING
#https://cran.r-project.org/web/views/Cluster.html
#Clustering w R np. K-means in two dimensions
mydata2kol <- mydata[c('engineSize', 'horsepower')]#choosing two parameters 
round(cor(mydata2kol), 2)                   #a big correlation
grupowanie_kmeans <- kmeans(mydata2kol, 3)  #3 different sets
plot(                                       #2D visualization with plot, abline, ade4 s.class
  mydata2kol,
  xaxt = 'n',
  yaxt = 'n',
  xlab = "X",
  ylab = "Y"
)
axis(1, pos = 0)
axis(2, pos = 0)
abline(v = 0, h = 0)
grupowanie_kmeans_cluster <- factor(grupowanie_kmeans$cluster)
# load 'ade4'
library(ade4)
s.class(
  mydata2kol,
  fac = grupowanie_kmeans_cluster,
  add.plot = TRUE,
  col = seq(1, nlevels(grupowanie_kmeans_cluster), 1)
)
aggregate(mydata2kol,by=list(grupowanie_kmeans$cluster),FUN=mean) # different groups, different group averages
groupk2 <- data.frame(mydata2kol, grupowanie_kmeans$cluster) 
Sys.sleep(2)                                # 2 second pause
library(cluster)
clusplot(mydata2kol, grupowanie_kmeans$cluster, color=TRUE, shade=TRUE,
         labels=2, lines=0)
Sys.sleep(2)                                # 2 second pause
library(fpc)
plotcluster(mydata2kol, grupowanie_kmeans$cluster) 
#library(fpc)                               # for comparisons
#cluster.stats(mydata, fit$cluster, fit2$cluster) 
Sys.sleep(2)                                # 2 second pause

#Clustering
#http://www.statmethods.net/advstats/cluster.html
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var)) #estimate a group number
for (i in 2:15) wss[i] <- sum(kmeans(mydata,centers=i)$withinss)
#$betweenss: group center distance square sums - have to be as large as possible
#$withinss: point distance sum within one cluster - have to be as small as possible
#$tot.withinss = sum ( $withinss )
#$totss = $tot.withinss + $betweenss
plot(1:15, wss, type="b", xlab="Group number", ylab="Sum of wss")
library(fpc)
pamk(mydata_scaled)$nc                      # estimated a clusters number on a scaled data
pamk(mydata)$nc                             # estimated a clusters number
Sys.sleep(2)                                # 2 second pause

#Clustering in R np. K-means in three dimensions
mydata3kol <- mydata[c('cityMpg','engineSize', 'horsepower')]      # choosing three parameters
mydata[c('cityMpg','engineSize', 'horsepower')]                  
grupowanie3_kmeans <- kmeans(mydata3kol, 3) # 3 clusters
grupowanie3_kmeans_cluster <- factor(grupowanie3_kmeans$cluster)
library(scatterplot3d)
scatterplot3d(mydata3kol,color=grupowanie3_kmeans_cluster,pch=19) #visualization in 3D 
library(rgl)
#http://www.sthda.com/english/wiki/a-complete-guide-to-3d-visualization-device-system-in-r-r-software-and-data-visualization
r3dDefaults$windowRect <- c(0,50, 800, 800) 
plot3d(mydata3kol, col=grupowanie3_kmeans_cluster, size = 10)     #interactive visualization in 3D 
Sys.sleep(2)                                # 2 second pause
dev.off()
#Clustering in R np. K-means in three dimensions based on PCA
#http://planspace.org/2013/02/03/pca-3d-visualization-and-clustering-in-r/
library(nFactors)
ev <- eigen(cor(mydata)) # get eigenvalues
ap <- parallel(subject=nrow(mydata),var=ncol(mydata),
               rep=100,cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS) 
Sys.sleep(2)                                # 2 second pause
library(FactoMineR)                         # PCA Variable Factor Map
result <- PCA(mydata)                       # graphs generated automatically 
plot(result)
Sys.sleep(2)                                # 2 second pause
pc <- princomp(mydata,cor=TRUE,scores=TRUE) # PCA calculate three artificial dimensions
summary(pc)
biplot(pc)
plot(pc,type="lines")
mdatapc<-pc$scores[,1:3]
str(mdatapc)
class(mdatapc)
grupowanie_pca_kmeans <- kmeans(mdatapc, 4) # 4 clusters
grupowanie_pca_kmeans_cluster <- factor(grupowanie_pca_kmeans$cluster)
scatterplot3d(mdatapc,color=grupowanie_pca_kmeans_cluster,pch=19)       #visualization in 3D 
r3dDefaults$windowRect <- c(0,50, 800, 800) 
plot3d(mdatapc, col=grupowanie_pca_kmeans_cluster, size = 10)           #interactive visualization in 3D 
text3d(pc$scores[,1:3],texts=rownames(mydata)) # adding chosen parameters
text3d(pc$loadings[,1:3], texts=rownames(pc$loadings), col="red")
coords <- NULL
for (i in 1:nrow(pc$loadings)) {
  coords <- rbind(coords, rbind(c(0,0,0),pc$loadings[i,1:3]))
}
lines3d(coords, col="red", lwd=4)
Sys.sleep(2)                                # 2 second pause

#Clustering in R: hierarchical groups
nc <- 3                                     # a cluster number
di <- dist(mydata, method="euclidean")
grupowanie_hclust <- hclust(di, method="ward.D2")
hcluster <-                                 # cut a tree to nc clusters
    as.factor((cutree(grupowanie_hclust,k=nc)-2)%%nc+1)
plot(grupowanie_hclust, xlab="")
rect.hclust(grupowanie_hclust, k=nc, border="red")
groupy <- cutree(grupowanie_hclust, k=nc)   # cut a tree to nc clusters
Sys.sleep(2)                                # 2 second pause

#Clustering Mclust
#https://cran.r-project.org/web/packages/mclust/vignettes/mclust.html
#mydata <- mydata_scaled
#library(mclust)
#grupowanie_mclust <- Mclust(mydata)
#plot(grupowanie_mclust, what = "BIC") 
#summary(grupowanie_mclust) 
#Sys.sleep(2)                               # 2 second pause


#Clustering in R: hierarchical groups pvclust 
#mydata <- mydata_scaled
#http://www.sigmath.es.osaka-u.ac.jp/shimo-lab/prog/pvclust/
library("pvclust")
grupowanie_pvclust <- pvclust(t(mydata), method.dist="euclidean", method.hclust="average", nboot=30)
plot(grupowanie_pvclust)
pvrect(grupowanie_pvclust, alpha=0.95) 
pvgroup <- pvpick(grupowanie_pvclust, alpha=0.95) 
#mydata$grupa <- NA
#for( i in seq_len(length(pvgroup$clusters))) mydata[pvgroup$clusters[[i]],]$grupa = i
#mydata$grupa


###################################################################################################
#DECISION TREES
#a new column with hclust grouping labels to learn classification 
#testing on the rest of data
nc <- 3                                      
grupy <- cutree(grupowanie_hclust, k=nc)    # cut a tree to nc clusters
mydata$grupa <- as.numeric(grupy)
mydata$grupa <- factor(mydata$grupa)             
# making training and testing sets
library(dplyr)
training_set <- sample_n(mydata,180)        # a sample set to train
testing_set <- mydata[-as.numeric(rownames(training_set)),]
clusterlabels <-testing_set$grupa #testing element labels
testing_set$grupa <- NULL                 # removing labels for test
rownames(training_set)<-seq_len(nrow(training_set))# rename rows
rownames(testing_set)<-seq_len(nrow(testing_set))# rename rows 

#Decision trees rpart
#http://www.statmethods.net/advstats/cart.html
#http://machinelearningmastery.com/non-linear-regression-in-r-with-decision-trees/
library(rpart)
#?rpart.control                             # set parameters of rpart
myclassifier_rpart <- rpart(grupa ~ ., method="class", data=training_set, minsplit=2)
#myclassifier_rpart <- rpart(grupa ~ ., method="anova", data=mydata)
printcp(myclassifier_rpart)                 # results
plotcp(myclassifier_rpart)                  # crossvalidation visualization
summary(myclassifier_rpart)                 # summary
# create a scheme picture
par(mar=c(0,5,3,5))
plot(myclassifier_rpart, uniform=TRUE,
     main="Decision tree for mtcars with the label fit (pvclust)")
text(myclassifier_rpart, use.n=TRUE, all=TRUE, cex=.8)
#write the picture to a file
post(myclassifier_rpart, file = "treerpart1.pdf",
     title = "Decision tree for mtcars with the label fit (pvclust)")
pred_labels1 <- predict(myclassifier_rpart, testing_set)   # predict labels
pred_labels1

#Decision tree ctree
library(party)
myclassifier_ctree <- ctree(grupa ~ ., data=training_set, controls = 
    ctree_control(mincriterion = 0,minbucket = 0,minsplit = 0,maxdepth = 100,savesplitstats = TRUE))
plot(myclassifier_ctree)
pdf('treec.pdf')
plot(myclassifier_ctree)
dev.off()
#plot(myclassifier_ctree, type="simple")
pred_labels2 <- predict(myclassifier_ctree, testing_set)    # predict labels
pred_labels2


#Decision tree randomForest - many trees and voting for solution 
#various numbers of parameters for every tree
library(RRF)
myclassifier_rrf <- RRF(grupa ~ ., data=training_set)
print(myclassifier_rrf)                     # show classification outcome
#summary(myclassifier_rrf)
importance(myclassifier_rrf)                # importance of each predictor 
pred_labels3 <- predict(myclassifier_rrf, testing_set)# predict labels
pred_labels3


#Mixture Discriminant Analysis
library(mda)
myclassifier_mda <- mda(grupa ~ ., data=training_set)
print(myclassifier_mda)                     # show classification outcome
summary(myclassifier_mda)
pred_labels4 <- predict(myclassifier_mda, testing_set)# predict labels
pred_labels4


#Regularized Discriminant Analysis
library(klaR)
myclassifier_rda <- rda(grupa ~ ., data=training_set)
print(myclassifier_rda)                     # show classification outcome
summary(myclassifier_rda)
pred_labels5 <- predict(myclassifier_rda, testing_set)   # predict labels
pred_labels5


#Gradient Boosted Machine 
#http://www.listendata.com/2015/07/gbm-boosted-models-tuning-parameters.html
library(gbm)
myclassifier_gbm <- gbm(grupa ~ ., data=training_set, distribution="gaussian", 
              bag.fraction = 0.5, n.trees = 1000, interaction.depth =6, 
              shrinkage = 0.1, n.minobsinnode = 1)
print(myclassifier_gbm)                     # show classification outcome
summary(myclassifier_gbm)
pred_labels6 <- predict(myclassifier_gbm, testing_set,n.trees = 10)   # predict labels
round(pred_labels6)



#Functions not tested written in the form function(package) among others:
#fda(mda), kernlab(ksvm), knn3(caret), naiveBayes(e1071), nnet(nnet), qda(MASS)
#J48(RWeka), PART(RWeka), C5.0(C50), vglm(VGAM), lda(MASS), plsda(caret)
#earth(earth), knnreg(caret), glmnet(glmnet), lars(lars), glmnet(glmnet)
#pcr(pls), plsr(pls)
#and many others: http://topepo.github.io/caret/modelList.html
#http://machinelearningmastery.com/how-to-get-started-with-machine-learning-algorithms-in-r/
#R webinars
#https://cran.r-project.org/web/packages/RSelenium/vignettes/OCRUG-webinar.html
#https://vimeo.com/89562453


#ATTENTION! Various labels for the testing set were sometimes obtained from different functions despite the same input data 
Można zauważyć, że wyniki etykietowania zbioru z nieznaną wartością fit z pvclust
fit1=c();for(i in 1:nrow(pred_labels1))fit1=c(fit1,which(pred_labels1[i,]==1))
separatelibrarylabels<-t(data.frame(rpart=as.numeric(fit1)))
separatelibrarylabels<-rbind(separatelibrarylabels,ctree=as.numeric(pred_labels2))
separatelibrarylabels<-rbind(separatelibrarylabels,rrf=as.numeric(pred_labels3))
separatelibrarylabels<-rbind(separatelibrarylabels,mda=as.numeric(pred_labels4))
separatelibrarylabels<-rbind(separatelibrarylabels,rda=as.numeric(pred_labels5$class))
separatelibrarylabels<-rbind(separatelibrarylabels,gbm=round(pred_labels6))
separatelibrarylabels

#########################################################################################################################
#CARET - Classifier General Wrapper about 170 classifiers
#https://cran.r-project.org/web/packages/caret/vignettes/caret.pdf
#170 classifiers list
#http://topepo.github.io/caret/modelList.html
#http://topepo.github.io/caret/bytag.html
#an exemplary model
#http://topepo.github.io/caret/training.html
#https://www.youtube.com/watch?v=7Jbb2ItbTC4 #caretwebinar
#http://www.r-bloggers.com/caret-webinar-materials/
library(caret)

#Decision tree rpart within the package caret
#chosen from 170 classifiers
#http://topepo.github.io/caret/modelList.html
#http://www.statmethods.net/advstats/cart.html
#http://machinelearningmastery.com/non-linear-regression-in-r-with-decision-trees/
library(rpart)
#myclassifier_rpart <- rpart(grupa ~ ., method="class", data=training_set, minsplit=2)
#rpart is now an argument of the function train from the package caret, we can only change parameters of the structure cp
myclassifier <- train(grupa ~ ., data=training_set,method = "rpart")
print(myclassifier)                         # show classification outcome
#summary(myclassifier)   
labels <- predict(myclassifier, testing_set)# predict labels
caretlibrarylabels<-t(data.frame(rpart=as.numeric(labels)))


#Decision tree ctree within the package caret
#chosen from 170 classifiers
#http://topepo.github.io/caret/modelList.html
library(party)
#ctree <- ctree(grupa ~ ., data=training_set, controls = 
#                 ctree_control(mincriterion = 0,minbucket = 0,minsplit = 0,maxdepth = 100,savesplitstats = TRUE))
#ctree is now an argument of the function train from the package caret, we can only change parameters of the structure mincriterion
myclassifier <- train(grupa ~ ., data=training_set,method = "ctree")
plot(myclassifier)
labels <- predict(myclassifier, testing_set)# predict labels
caretlibrarylabels<-rbind(caretlibrarylabels,t(data.frame(ctree=as.numeric(labels))))


#Decision tree randomForest RRF (many trees and voting for solution -
#various numbers of parameters for every tree) within the package caret
#chosen from 170 classifiers
#http://topepo.github.io/caret/modelList.html
#myclassifier_rrf <- RRF(grupa ~ ., data=training_set)
#RRF is now an argument of the function train from the package caret, we can only change parameters mtry, coefReg, coefImp
myclassifier <- train(grupa ~ ., data=training_set,method = "RRF")
print(myclassifier)                         # show classification outcome
#summary(myclassifier)
labels <- predict(myclassifier, testing_set)# predict labels
caretlibrarylabels<-rbind(caretlibrarylabels,t(data.frame(rrf=as.numeric(labels))))



#Mixture Discriminant Analysis within the package caret
#chosen from 170 classifiers
#http://topepo.github.io/caret/modelList.html
library(mda)
#myclassifier_mda <- mda(grupa ~ ., data=training_set)
#mda is now an argument of the function train from the package caret, we can only change parameters subclasses
myclassifier <- train(grupa ~ ., data=training_set,method = "mda")
print(myclassifier)                         # show classification outcome
summary(myclassifier)
labels <- predict(myclassifier, testing_set)# predict labels
caretlibrarylabels<-rbind(caretlibrarylabels,t(data.frame(mda=as.numeric(labels))))


#Regularized Discriminant Analysis within the package caret
#chosen from 170 classifiers
#http://topepo.github.io/caret/modelList.html
library(klaR)
#myclassifier_rda <- rda(grupa ~ ., data=training_set)
#rda is now an argument of the function train from the package caret, we can only change parameters gamma, lambda
myclassifier <- train(grupa ~ ., data=training_set,method = "rda")
print(myclassifier)                         # show classification outcome
summary(myclassifier)
labels <- predict(myclassifier, testing_set)# predict labels
caretlibrarylabels<-rbind(caretlibrarylabels,t(data.frame(rda=as.numeric(labels))))


#Gradient Boosted Machine within the package caret
#chosen from 170 classifiers
#http://topepo.github.io/caret/modelList.html
#http://www.listendata.com/2015/07/gbm-boosted-models-tuning-parameters.html
#with the HELP of trainControl can be add k-fold crossvalidation
https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation
#3-fold crossvalidation with average outcome repeated two times
ctrl <- trainControl(method = "repeatedcv", number = 3, repeats = 2, 
                     classProbs = FALSE)

#error I  (false positive - healthy seen as ill) - true null hypothesis is rejected
#error II (false negative) - false null hypothesis is accepted
#error III(overfitting to the training set) 
#crossvalidation stops the error III

gbmGrid <-  expand.grid(interaction.depth = 9, 
                        n.trees = 100, shrinkage = 0.1, n.minobsinnode = 2)
myclassifier <- train(grupa ~ ., data=training_set,
                  method = "gbm",
                  verbose = FALSE,
                  trControl = ctrl,
                  tuneGrid = gbmGrid)
print(myclassifier)                         # show classification outcome
summary(myclassifier)
labels <- predict(myclassifier, testing_set,n.trees = 10) # predict labels
caretlibrarylabels<-rbind(caretlibrarylabels,t(data.frame(gbm=as.numeric(labels))))
#trellis.par.set(caretTheme())
#plot(traingbm)
#ggplot(traingbm)

#ATTENTION! Various labels for the testing set were sometimes obtained from different functions despite the same input data 
#different random samples and method parameters can influence this situation
separatelibrarylabels
#labels obtained with the help of the carret package
caretlibrarylabels
#the true labels obtained in clustering removed for testing
t(data.frame(spr=as.numeric(clusterlabels)))
#########################################################################################################################
Sys.sleep(5)                                # 5 second pause


#########################################################################################################################
#TIME SERIES - predict trends, make forecast, fit to the model
#http://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/
#http://a-little-book-of-r-for-time-series.readthedocs.io/en/latest/src/timeseries.html
#https://www.otexts.org/fpp/8/9
ndata <- "AirPassengers"                    #mydata passenger numbers per month during several years
data(list=ndata)
mydata<-get(ndata)
mydata                                      # a given set
start(mydata)                               # beginning date
end(mydata)                                 # finishing end
frequency(mydata)                           # a number of data in a time period
time(mydata)                                # common vector of all data points
cycle(mydata)                               # cycle 
summary(mydata)                             # summary averages
plot(as.vector(time(mydata)), as.vector(mydata), type = "l")# plot
plot(mydata)                          
plot(log(mydata))
abline(reg=lm(mydata~time(mydata)))         # regression line
plot(aggregate(mydata,FUN=mean))            # year mean
boxplot(mydata~cycle(mydata))               # season effect - regular trends
mydata_decomp<-decompose(mydata)            # decomposition on regular and other trends
str(mydata_decomp)
plot(mydata_decomp)                            # plots of various trends
plot(mydata - mydata_decomp[['seasonal']])     # mydata without regular trends

library(tseries)
adf.test(mydata, "stationary", k=0)            #Augmented Dickey-Fuller Test 
adf.test(log(mydata), "stationary", k=0)       #does not work
adf.test(diff(mydata), "stationary", k=0)      #it should be tests from fUnitRoots
adf.test(diff(log(mydata)), "stationary", k=0) #it does not work!

library(fUnitRoots);
adfTest(mydata);                               # ADF Test for p<0.01  
adfTest(log(mydata));                          # for the arima model
adfTest(diff(mydata));                         # for difference between data
adfTest(diff(log(mydata)));                    # the best results for differences on log(data)
acf(mydata)                                    # Total Correlation Chart (Auto – correlation Function / ACF) 
                                               # for successive corelations  x(t) z  x(t-1) , x(t-2)
pacf(mydata)                                   # partial correlation function (PACF) below the line AR
acf(log(mydata))                               # acf of log
pacf(log(mydata))                              # pacf of log
acf(diff(mydata))                              # acf of difference
pacf(diff(mydata))                             # pacf of difference
acf(diff(log(mydata)))                         # acf of log difference 
pacf(diff(log(mydata)))                        # pacf of log difference 
acf(diff(mydata,diff=2))                       # acf of second level difference 
pacf(diff(mydata,diff=2))                      # pacf of second level difference
acf(diff(log(mydata),diff=2))                  # acf of log second level difference
pacf(diff(log(mydata),diff=2))                 # pacf of log second level difference

library(forecast)
tsdisplay(mydata)                              # tsdisplay shows acf and pacf 
tsdisplay(log(mydata));                        # peak outliners
tsdisplay(diff(mydata,12));                    # shows PACF AR components
tsdisplay(diff(log(mydata,12)));               # w ACF MA components MA(1) the last such file
                                            

#for Auto-Regressive (AR) PACF decreases for x line difference x -  x(t) to x(t+N) N>>0
#for Moving Average (MA) ACF for irregular trends it decreases - x(t) to x(t+N) N~1
#c(p,d,q) p connected with AR oraz q connected with MA, d it is a level diff
#seasonal option
#the best tsdisplay(forecast(auto.arima(mydata),20)$residuals) from the package forecast check and add AR and MV 
library(forecast)
model_autoarima<-auto.arima(mydata)            # ARIMA model 
model_autoarima                                # automatically obtained parameters from ACF and PACF 
myforecast<-forecast(model_autoarima,h=20)
plot(myforecast)                               # plot for next 20 months with invcreasing prediction error
Sys.sleep(2)                                   # 2 second pause
tsdisplay(myforecast$residuals)                # show error analysis, in ACF a significant peak on 23rd position MA(11)
# it changes a model ARIMA(0,1,1)(0,1,0)[12] to ARIMA(0,1,1)(0,1,11)[12]
Box.test(myforecast$residuals, lag=12, type="Ljung-Box")
predict(model_autoarima,20)                    # 20 periods forward
#in model minimalize aic 
model_arima <- arima(log(mydata), c(0, 1, 1), seasonal = list(order = c(0, 1, 11), period = 12)) # two level difference
predyktor <- 
  predict(model_arima, n.ahead = 10*12)        # 10 years ahead prediction
ts.plot(mydata,2.718^predyktor$pred, log = "y", lty = c(1,3))#prediction plot
myforecast<-forecast.Arima(model_arima, h=24)  # myforecast of arima for 24 periods
plot(myforecast)                               # show a plot
acf(myforecast$residuals, lag.max=20)          # 
tsdisplay(myforecast$residuals)
# in Box.test lag = 2 * period 
Box.test(myforecast$residuals, lag=24, type="Ljung-Box")# for p<<0 autocorrelations - bad predictor for MA and for diff the better one
Sys.sleep(2)                                   # 2 second pause


#model_predykcji <- HoltWinters(mydata, beta=FALSE, gamma=FALSE)# gamma = FALSE nie okresowy model
#model_predykcji <- HoltWinters(mydata, beta=FALSE)  # beta=FALSE wygładanie exponens
#model_predykcji <- HoltWinters(mydata)        # 
model_predykcji <- HoltWinters(mydata, seasonal = "mult")# periodic "multiplicative", default additive
plot(model_predykcji)                          # red line for prediction
model_predykcji$SSE                            # prediction quality
myforecast<-forecast.HoltWinters(model_predykcjAi,h=8)# forecast for 8 periods
plot(myforecast)
acf(myforecast$residuals, lag.max=20)          # checking the forecast
Box.test(myforecast$residuals, lag=24, type="Ljung-Box")# for p<<0 autocorrelations - a bad predictor 
plot.ts(myforecast$residuals)                  # error plot with variance
plotForecastErrors<-function(forecasterrors)   # into normal distribution
{
  # error histogram 
  mybinsize <- IQR(forecasterrors)/4
  mysd   <- sd(forecasterrors)
  mymin  <- min(forecasterrors) - mysd*5
  mymax  <- max(forecasterrors) + mysd*3
  # normal distribution with teh mean equal to 0 and a standard deviation mysd
  mynorm <- rnorm(10000, mean=0, sd=mysd)
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 }
  if (mymax2 > mymax) { mymax <- mymax2 }
  # red error histogram with normal distribution
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  # for freq=FALSE field under histogram = 1
  # normal distribution with teh mean equal to 0 and a standard deviation mysd
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  # blue normal distribution 
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2)
}
plotForecastErrors(myforecast$residuals)       # is it similar to normal distribution

library(forecast)
model_predykcji=ets(mydata)                    # fully automatic prediction
plot(forecast(model_predykcji))                # plot with error margins 
Sys.sleep(2)                                   # 2 second pause
predict(model_predykcji,10)                        # forecast for 10 periods

#Neural networks in forecast 
model_predykcji <- nnetar(mydata)              # special neural networks
myforecast <- forecast(model_predykcji)        # myforecast  
plot(myforecast)                               # blue plot myforecast
Sys.sleep(2)                                   # 2 second pause
tsdisplay(myforecast$residuals)                # show error analysis
Box.test(myforecast$residuals, lag=12, type="Ljung-Box")

#time series
#http://www.r-bloggers.com/additive-modelling-global-temperature-time-series-revisited/
#http://www.r-bloggers.com/additive-modelling-and-the-hadcrut3v-global-mean-temperature-series-2/
#########################################################################################################################

