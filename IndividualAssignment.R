library(rSAFE)
head(HR_data)
install.packages('rSAFE')
install.packages('dplyr')
library(tidyr) 
install.packages('tidyr')
library(data.table) 
library(dplyr)
library(ggplot2)
install.packages('Information')
library(Information)
library(caret)
library(car)
library(mlr)        
library(pROC)        
library(e1071)       
library(gridExtra)   
library(nnet)  
library(pROC)          # AUC, ROC
library(tree)          # CART model
library(randomForest)  # Bagging and RF
library(gbm)           # Boosting tree 
install.packages('tree')
library(MASS)
library(tree) 
library(ROCR) 
library(Metrics)
install.packages("Metrics")

# Clear the R environment 
rm(list=ls()) 

str(HR_data)

# predicting whether an employee is likely to leave the company. 
# This will be my dependent variable - 1 to leave and 0 not to leave / binary classification 

# no missing values 
is.na(HR_data)

########## PART 1: Data Exploration ############

# Calculate employee churn rate in general
table(HR_data$left)
# of the 14 999 employees in total, 11 428 did not leave the company and 3 571 did leave the company  
HR_data %>%
  summarize(left = mean(left))
# The churn rate is 23,81% 

# Calculate churn rate per salary level 
df_salary <- HR_data %>%
  group_by(salary) %>%
  summarize(left = mean(left))
df_salary

ggplot(df_salary, aes(x = salary, y = left)) +
  geom_col() 

# Calculate employee churn rate per satisfaction level 
df_satisfaction_level <- HR_data %>%
  group_by(satisfaction_level) %>%
  summarize(left = mean(left))
df_satisfaction_level

# Calculate employee churn rate per number of projects 
df_number_project <- HR_data %>%
  group_by(number_project) %>%
  summarize(left = mean(left))
df_number_project

# Calculate employee churn rate per average monthly hour 
# Covert column from integer to numeric
HR_data$average_monthly_hour <- as.numeric(HR_data$average_monthly_hour)
str(HR_data)
df_average_monthly_hour <- HR_data %>%
  group_by(average_monthly_hour) %>%
  summarize(left = mean(left))
df_average_monthly_hour

# Calculate employee churn rate per time spend at the company
df_time_spend_company <- HR_data %>%
  group_by(time_spend_company) %>%
  summarize(left = mean(left))
df_time_spend_company



########## Part 2: Information value ###########

IV <- create_infotables(data = HR_data, y="left")
IV$Summary



########## PART 3: Splitting the data #########

set.seed(123)
train_index <- sample(1:nrow(HR_data), nrow(HR_data) * 0.7)
train <- HR_data[train_index, ]
test <- HR_data[-train_index, ]
dim(train)
dim(test)

#Baseline Accuracy
prop.table(table(HR_data$left))



########### PART 4: Cross-validation #######

# k-fold Cross Validation

set.seed(123)
# k = 10 
train_control <- trainControl(method = "cv",
                              number = 1)
# model = Naive Bayes 
nb_fit <- train(factor(left) ~., data=HR_data, method = "naive_bayes", trControl=train_control, tuneLenght = 0)
nb_fit



########### Part 5: Modelling ############## 

# A) Multiple logistic regression model 
multiplelog <- glm(left ~ ., family = "binomial", data = train)
summary(multiplelog)

# Fit the LDA model with 2 best predictors from LogRes model
library(MASS)
md_lda <- lda(left ~ satisfaction_level + number_project, data=train)
md_lda

plot(md_lda)

# Predict + AUC 
log_predict <- predict(multiplelog,newdata = test,type = "response")

pr <- prediction(log_predict,test$left)
perf <- performance(pr,measure = "tpr",x.measure = "fpr") 

plot(perf) > auc(test$left,log_predict)
auc(test$left,log_predict) # 0.8221


# B) Random forest 

library(randomForest)  
# Select variables (res_vars) for the model to predict 'left'
res_vars <- c("satisfaction_level", "last_evaluation", "number_project", "average_monthly_hour", "time_spend_company", "work_accident", "promotion_last_5years", "sales", "salary", "left")
set.seed(222)
HR_RF <- randomForest(left ~ .,
                                data = HR_data[res_vars],
                                ntree=500, importance = TRUE,
                                na.action = na.omit)
varImpPlot(HR_RF,type=1,
           main="Variable Importance (Accuracy)",
           sub = "Random Forest Model")

var_importance <-importance(HR_RF)
var_importance # view results 

# Confusion matrix
RF_pred <- predict(HR_RF, newdata = test)
predict(HR_RF, newdata = test)


predn = ifelse(RF_pred > 0.50, 1, 0)
confusionMatrix(as.factor(predn), as.factor(test$left))


# C) Gradient Boosting 

require(gbm)
HR.boost = gbm(left ~ ., data=train, distribution = "gaussian",n.trees = 10000,
               shrinkage = 0.01, interaction.depth = 4) 
HR.boost
summary(HR.boost)

# correlation 
# negative correlation 
cor(HR_data$satisfaction_level,HR_data$left)

# Predict on test set 
n.trees = seq(from=100 ,to=10000, by=100)

predmatrix<-predict(HR.boost,train,n.trees = n.trees)
dim(predmatrix)

#Calculating The Mean squared Test Error
test.error<-with(train,apply((predmatrix-left)^2,2,mean))
head(test.error)

#Plotting the test error vs number of trees
plot(n.trees , test.error , pch=19,col="blue",xlab="Number of Trees",ylab="Test Error", main = "Perfomance of Boosting on Test Set")

# Evaluation - AUC score 
preds <- predict(HR.boost, newdata = test, n.trees = 70)
labels <- test[,"left"]
cvAUC::AUC(predictions = preds, labels = labels)
install.packages('cvAUC')


# D) Support Vector Machine 

# Fit Support Vector Machine model to data set
classifier = svm(formula = left ~ .,
                 data = train,
                 type = 'C-classification',
                 kernel = 'linear')

y_pred = predict(classifier, newdata = test)
y_train_pred = predict(classifier, newdata = train)

# AUC 
cm = table(test$left, y_pred)
cm
cm2 = table(train$left, y_train_pred)
cm2


