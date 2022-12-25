# Tan Shu Fei 7020466
# INFO411 A2 2022
#--------------------------------------------------------------
# DATA PREPROCESSING

#import libraries
library(e1071)
library(randomForest)
library(rpart)
library(ROCR)
library(RSNNS)
library(partykit)

# Load the data into a data frame
cw_all<- read.csv("./creditworthiness.csv")

# Shuffle the dataset
shuffleIndex = sample(1:nrow(cw_all))
head(shuffleIndex)
cw_all = cw_all[shuffleIndex,]

# Separate records with NO credit rating
# E.g. credit_rating = 0
cw_known <- subset(cw_all, credit.rating > 0) # If not empty
cw_unknown <- subset(cw_all, credit.rating == 0) # If empty

#Split data set into train & test set
cw_train <- cw_known[1:(nrow(cw_known) / 2),] # First half
cw_test <- cw_known[-(1:(nrow(cw_known) / 2)),] # Second half

# Convert char data type to factor
cw_train$credit.rating <- factor(cw_train$credit.rating) 
cw_test$credit.rating <- factor(cw_test$credit.rating)
#--------------------------------------------------------------
# Fit a Decision Tree

# QUESTION 2A) Report the resulting tree.
(cw_rpart = rpart(credit.rating~. , data = cw_train)) # Print decision tree
plot(cw_rpart) # Plot decision tree
text(cw_rpart) # Show the labels

# Create a decision tree with ctree
(cw_ctree <- ctree(credit.rating ~ ., data = cw_train)) # Print decision tree
plot(cw_ctree) # Plot decision tree
#--------------------------------------------------------------
# QUESTION 2B) Predict the credit rating of a hypothetical “median”

# Data in hypothetical customer.
cust_data <-  c(0,1,1,0,3,
                0,3,3,0,3,
                3,3,3,3,3,
                3,3,3,3,3,
                3,3,3,3,3,
                3,3,3,3,3,
                3,3,3,3,3,
                3,3,3,3,3,
                3,3,3,3,3 )
median_cust = data.frame() # Create an empty dataframe
median_cust = rbind(median_cust, cust_data) # Pass in datafram + customer data
colnames(median_cust) = names(cw_known)[-46] # Remove credit.rating

(cust_pred = predict(cw_rpart, median_cust)) # Predict DT + median customer
#--------------------------------------------------------------
# QUESTION 2C) Produce the confusion matrix on test set + accuracy rate
predictTest <- predict(cw_rpart, cw_test) # Predict DT + test set
(dt_confusionMatrix <- confusionMatrix(cw_test$credit.rating, predictTest))# C.M
sum(diag(dt_confusionMatrix))/sum(dt_confusionMatrix) # Get accuracy rate 
#--------------------------------------------------------------
# QUESTION 2D) Numerical value of entropy corresponding to 
# the 1st split at the top of the tree

(beforeCountFreq = table(cw_train$credit.rating)) #Count all classes in credit.rating
(beforeClassProb = beforeCountFreq/sum(beforeCountFreq)) # Find prob of each class
(beforeEntropy = -sum(beforeClassProb * log2(beforeClassProb))) # Cal entropy (b4 split)

# functionary == 0
(countFreq0 = table(cw_train$credit.rating[cw_train$functionary == 0]))
(classProb0 = countFreq0/sum(countFreq0)) 
(functionaryEnt0 = -sum(classProb0 * log2(classProb0))) # Cal entropy

# functionary == 1
countFreq1 = table(cw_train$credit.rating[cw_train$functionary == 1]) 
classProb1 = countFreq1/sum(countFreq1)
(functionaryEnt1 = -sum(classProb1 * log2(classProb1))) # Cal entropy

(ent = (beforeEntropy - (functionaryEnt0 * sum(countFreq0) + 
                         functionaryEnt1 * sum(countFreq1)) / 
                         sum(sum(countFreq0) + sum(countFreq1))) ) # Total entropy
#--------------------------------------------------------------
# QUESTION 2E) Fit a random forest model to the training set 
rftune_train <- randomForest(factor(credit.rating)~., data = cw_train, 
                             mtry = 24, ntree=500, 
                             stepFactor=2, improve=0.2) # Fit RFT aft tuning
print(rftune_train) # print summary + confusion matrix
plot(rftune_train) # Generate plot
#--------------------------------------------------------------
# QUESTION 2F) Produce the confusion matrix on test set + accuracy rate
rftune_pred = predict(rftune_train, cw_test[,-46]) # Predict RFT + test set
rfconfusion_tune = confusionMatrix(cw_test$credit.rating, rftune_pred) # C.M
rfconfusion_tune # Generate confusion matrix
sum(diag(rfconfusion_tune))/sum(rfconfusion_tune) # Get accuracy rate 
#-------------------------------------------------------------------------------
# Fit with SVM Model
#-------------------------------------------------------------------------------
svm_model = svm(credit.rating ~ ., data = cw_train, kernel = "radial")

# QUESTION 3A) Predict the credit rating of a hypothetical “median” customer
(svm_pred <- predict(svm_model, median_cust, decision.values = TRUE))

# QUESTION 3B) Produce the confusion matrix w test set + accuracy rate
svm_pred = predict(svm_model, cw_test[,-46]) # Predict svm + test set
(svmconfusion_tune = confusionMatrix(cw_test$credit.rating, svm_pred)) # C.M
sum(diag(svmconfusion_tune))/sum(svmconfusion_tune) # Get accuracy rate 

# QUESTION 3C) Tune the SVM to improve prediction
summary(tune.svm(credit.rating ~ ., data = cw_train, 
                 gamma = 10^(-4:-1), cost = 10^(0:2))) # Get summary
svm_tuned = svm(credit.rating ~ ., data = cw_train,
                cost=1, gamma = 0.01) # Fit TUNED SVM Model
svm_tunedpred = predict(svm_tuned, cw_test[,-46]) # Predict values w test set
(confusionTunedSVM = confusionMatrix(cw_test$credit.rating, svm_tunedpred)) # C.M
sum(diag(confusionTunedSVM))/sum(confusionTunedSVM) # Get accuracy rate 
#--------------------------------------------------------------
# Fit with Naive Bayes Model
nb_model <- naiveBayes(credit.rating ~ ., data = cw_train, type='raw')

# QUESTION 4A) Predict the credit rating of a hypothetical “median” customer
(nb_pred <- predict(nb_model, median_cust, type='class'))
(nb_pred <- predict(nb_model, median_cust, type='raw'))

# QUESTION 4B) Reproduce the first 20 lines for the Naive Bayes fit
nb_tunepred = predict(nb_model, cw_test[,-46]) # Predict values w test set
(nb_confusion = with(cw_test, table(nb_tunepred, credit.rating))) # C.M
sum(diag(nb_confusion))/sum(nb_confusion) # Get accuracy rate 
#--------------------------------------------------------------
# Fit with Logistic Regression Model

# QUESTION 6A) Fit a LR model to predict whether a customer gets credit.rating of A
lr_model <- glm((credit.rating==1) ~ ., data = cw_train, family=binomial)
anova(lr_model)

# QUESTION 6B) Report the summary table
summary(lr_model)

# QUESTION 6D) Fit an SVM model of your choice to the training set.
# Fit an SVM model of your choice to the training set 
summary(tune.svm(credit.rating ~ ., data = cw_train, 
                 gamma = 10^(-4:-1), cost = 10^(0:2))) # Get summary
(svm2 = svm(I(credit.rating == 1)~ ., data = cw_train, 
            cost = 1, gamma = 0.01, type = "C")) # Tune SVM Model

# QUESTION 6E)
svmfitpred = predict(svm2, cw_test[,-46], decision.values =TRUE)
glmfitpred = predict(lr_model, cw_test[,-46]) 

confusionSVM2 <- prediction(-attr(svmfitpred, "decision.values"), cw_test$credit.rating == 1)
confusion_lr = prediction(glmfitpred, cw_test$credit.rating == 1)

# QUESTION 6E) Produce an ROC chart comparing the LR & SVM from the test set
svm_roc <- performance(confusionSVM2, "tpr", "fpr") # Create ROC Curve
lr_roc <- performance(confusion_lr, "tpr", "fpr") # Create ROC Curve

# Plot the graph
plot(lr_roc, col=1) # Plot for logistic regression
plot(svm_roc, col= 2 ,add=TRUE) # Plot for SVM
abline(0, 1, lty = 3) # Middle line
legend(0.8, 0.2, c('glm','svm'), 1:2) # Add legend to the plot
