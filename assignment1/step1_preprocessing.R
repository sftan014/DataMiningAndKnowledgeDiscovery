# Preprocessing

### LOAD LIBRARIES - install with:
#install.packages(c("kohonen", "dummies", "ggplot2", "maptools", "sp", "reshape2", "rgeos"))
library(kohonen)
library(dummies)
library(ggplot2)
library(sp)
library(maptools)
library(reshape2)
library(rgeos)

# Colour palette definition
pretty_palette <- c("#1f77b4", '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2')

### DATA PREPARATION

# Census data comes in counts of people per area. 
# To compare areas, we will convert a number of the
# stats collected into percentages. Without this, 
# the primary differentiator between the different 
# areas would be population size.

# options to explore a few different map types:
small_areas <- TRUE
filter <- TRUE

# Load the data into a data frame
# Get the map of these areas and filter for Dublin areas.

data_raw <- read.csv("./creditworthiness.csv")  

#--------------------------------------------------------------
# Remove any values with int 0 in attribute 'credit rating'
row_46 <- data_raw[, c(46)]

##Go through each row and determine if a value is zero
row_sub = apply(data_raw[c(46)], 1, function(row) all(row !=0))

##Subset as usual
data_raw <- data_raw[row_sub,]
#--------------------------------------------------------------
# Calculate the correlation of each attribute with credit rating
#for (i in 1:45) {
#  print(cor(data_raw[46], data_raw[i]))
#}