library(RSNNS)

# Load dataset
fullDataSet <- read.csv("creditworthiness.csv")

# Select all entries for which the credit rating is known
# Check if data NOT empty
knownData <- subset(fullDataSet, fullDataSet[,46] > 0)

# Select all entries for which the credit rating is unknown
# Check if data is null/empty/0
unknownData <- subset(fullDataSet, fullDataSet[,46] == 0)

#separate value from targets
trainValues <- knownData[,1:45]
trainTargets <- decodeClassLabels(knownData[,46])
unknownsValues <- unknownData[,1:45]

# Split data set into training and test set
trainset <- splitForTrainingAndTest(trainValues, trainTargets, ratio=0.2)
trainset <- normTrainingAndTestSet(trainset)

# Train the Machine Learning Program model on the trining subset & validation set
model <- mlp(trainset$inputsTrain, trainset$targetsTrain, size=5, learnFuncParams=c(0.01), maxit=250, inputsTest=trainset$inputsTest, targetsTest=trainset$targetsTest)
predictTestSet <- predict(model,trainset$inputsTest)

# Tune the model -> Prints prediction model
confusionMatrix(trainset$targetsTrain,fitted.values(model))
confusionMatrix(trainset$targetsTest,predictTestSet)

# Prints the 4 plots
par(mfrow=c(2,2))
plotIterativeError(model)
plotRegressionError(predictTestSet[,2], trainset$targetsTest[,2])
plotROC(fitted.values(model)[,2], trainset$targetsTrain[,2])
plotROC(predictTestSet[,2], trainset$targetsTest[,2])
 
# Confusion matrix with 402040-method
confusionMatrix(trainset$targetsTrain, encodeClassLabels(fitted.values(model),method="402040", l=0.4, h=0.6))

# Show detailed information of the model
summary(model)
model
weightMatrix(model)
extractNetInfo(model)
 
