
# ------------------- SOM TRAINING ---------------------------

#choose the variables with which to train the SOM
#the following selects column 2,4,5,8
data_train <- data_raw[, c(1, 2, 3, 6, 12)] 
data_raw <- row_46[complete.cases(row_46),]

# now train the SOM using the Kohonen method
data_train_matrix <- as.matrix(scale(data_train))
names(data_train_matrix) <- names(data_train)
require(kohonen)
x_dim=10
y_dim=10

if (small_areas){
  # larger grid for the small areas example (more samples)
  som_grid <- somgrid(xdim = x_dim, ydim=y_dim, topo="hexagonal")  
} else {
  som_grid <- somgrid(xdim = x_dim/2, ydim=y_dim/2, topo="hexagonal")  
}

# Train the SOM model!
if (packageVersion("kohonen") < 3){
  system.time(som_model <- som(data_train_matrix, 
                             grid=som_grid, 
                             rlen=1000, 
                             alpha=c(0.5,0.01),
                             n.hood = "circular",
                             keep.data = TRUE ))
}else{
  system.time(som_model <- som(data_train_matrix, 
                             grid=som_grid, 
                             rlen=1000, 
                             alpha=c(0.5,0.01),
                             mode="online",
                             normalizeDataLayers=false,
                             keep.data = TRUE ))
}
summary(som_model)

# Remove the variables after using
rm(som_grid, data_train_matrix)

