install.packages("readr")
install.packages("ggplot2")
install.packages("rpart.plot")
install.packages("caret")
install.packages("tidyverse")
install.packages("factoextra")

library(readr)
library(ggplot2)
library(rpart.plot)
library(dplyr)
library(caret)
library(pROC)
library(tidyverse)
library(cluster)
library(factoextra)


#lode the data set
data<-read.csv("C:\\Users\\shard\\Desktop\\Ishara\\MY 2ed year\\End exam\\R\\Final project\\HeartDiseaseTrain-Test.csv")
#Summary of data
str(data)
summary(data)
head(data)




# Identify missing values
missing <- colSums(is.na(data))
print(missing)

# Remove rows with missing values
data_clean <- na.omit(data)
print(data_clean)




# Preprocessing the data
# Convert categorical variables to factors
data$sex <- as.factor(data$sex)
data$chest_pain_type <- as.factor(data$chest_pain_type)
data$fasting_blood_sugar <- as.factor(data$fasting_blood_sugar)
data$rest_ecg <- as.factor(data$rest_ecg)
data$exercise_induced_angina <- as.factor(data$exercise_induced_angina)
data$slope <- as.factor(data$slope)
data$vessels_colored_by_flourosopy <- as.factor(data$vessels_colored_by_flourosopy)
data$thalassemia <- as.factor(data$thalassemia)
data$target <- as.factor(data$target)




# Split the dataset into training 
set.seed(123)
train_index <- createDataPartition(data$target, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]


# Display the dimensions of the training and testing sets
dim(train_data)
dim(test_data)





# Visualizations 

age_plot <- ggplot(data_clean, aes(x = age)) + 
  geom_histogram(bins = 10, fill = 'blue', color = 'black') + 
  ggtitle('Distribution of Age')
print(age_plot)

gender_plot <- ggplot(data_clean, aes(x = sex, fill = sex)) + 
  geom_bar(color = 'black') + 
  ggtitle('Gender Distribution') + 
  scale_fill_manual(values = c("Female" = "lightpink", "Male" = "lightblue")) + 
  theme_minimal()
print(gender_plot)

cp_plot <- ggplot(data_clean, aes(x = chest_pain_type, fill = chest_pain_type)) + 
  geom_bar() + 
  ggtitle('Frequency of Chest Pain Types') + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(cp_plot)

# Box Plot for Age 
ggplot(data_clean, aes(x = factor(target), y = age, fill = factor(target))) +
  geom_boxplot() +
  labs(title = 'Age by Heart Disease Status', x = 'Heart Disease Status', y = 'Age') +
  theme_minimal() +
  theme(plot.background = element_rect(fill = 'lightyellow'))


#Prediction Decision Tree
fit <- rpart(target ~ ., data = train_data, method = 'class')
rpart.plot(fit, main = 'Decision Tree for Heart Disease Prediction'
           , extra = 106, under = TRUE, faclen = 0)





# Classification Models
# Logistic Regression
logistic_model <- train(target ~ ., data = train_data, 
                        method = "glm", family = "binomial")

# Decision Tree
decision_tree_model <- train(target ~ .,
                             data = train_data, 
                             method = "rpart")

 

# Evaluate the models on the test set
logistic_pred <- predict(logistic_model, 
                         newdata = test_data)
decision_tree_pred <- predict(decision_tree_model,
                              newdata = test_data)
random_forest_pred <- predict(random_forest_model,
                              newdata = test_data)

# Model Accuracies
logistic_confusion <- confusionMatrix(logistic_pred, test_data$target)
decision_tree_confusion <- confusionMatrix(decision_tree_pred,
                                           test_data$target)
random_forest_confusion <- confusionMatrix(random_forest_pred,
                                           test_data$target)

model_accuracies <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(logistic_confusion$overall["Accuracy"],
               decision_tree_confusion$overall["Accuracy"],
               random_forest_confusion$overall["Accuracy"])
)
print(model_accuracies)



# Confusion Matrices
logistic_cm <- as.data.frame(logistic_confusion$table)
decision_tree_cm <- as.data.frame(decision_tree_confusion$table)
random_forest_cm <- as.data.frame(random_forest_confusion$table)

logistic_cm$Model <- "Logistic Regression"
decision_tree_cm$Model <- "Decision Tree"
random_forest_cm$Model <- "Random Forest"

combined_cm <- rbind(logistic_cm, decision_tree_cm, 
                     random_forest_cm)

ggplot(combined_cm, aes(x = Reference, y = Prediction,
                        fill = Freq)) + 
  geom_tile(color = "white") + 
  facet_wrap(~ Model) + 
  scale_fill_gradient(low = "white", high = "blue") + 
  labs(title = "Confusion Matrices for Classification Models", 
       x = "Actual", y = "Predicted") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, 
                                   hjust = 1))





# Clustering Models

# K-means Clustering
set.seed(123)
wss <- sapply(1:10, function(k) {kmeans(scaled_data,
                                        centers = k, 
                                        nstart = 25)$tot.withinss})
numeric_data <- data_clean %>% select_if(is.numeric)
scaled_data <- scale(numeric_data)
print(scaled_data)


# Determine the optimal number of clusters using the Elbow method
optimal_k <- 3  
kmeans_result <- kmeans(scaled_data, centers = optimal_k, nstart = 25)
plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters (k)",
     ylab = "Total within-cluster sum of squares (WSS)",
     main = "Elbow Method for Optimal k")


# Visualizing Clusters
cluster_plot <- fviz_cluster(kmeans_result, data = scaled_data,
                             geom = "point",
                             ellipse.type = "convex", 
                             ggtheme = theme_bw()
) +
  labs(title = "K-means Clustering of Heart Disease Data")
print(cluster_plot)

# Add cluster assignments to the original data
data_clean$cluster <- kmeans_result$cluster

# Summary Statistics for Clusters
cluster_summary <- aggregate(data_clean[, c("age", "cholestoral", 
                                            "resting_blood_pressure", 
                                            "Max_heart_rate")], 
                             by = list(Cluster = data_clean$cluster), 
                             FUN = mean)
print(cluster_summary)



# Count the number of observations in each cluster
cluster_counts <- table(data_clean$cluster)
print(cluster_counts)



# Scatter plot for clustering results
ggplot(data_clean, aes(x = age, y = cholestoral,
                       color = factor(cluster))) +
  geom_point(alpha = 0.6) +
  scale_color_discrete(name = "Cluster") +
  labs(title = "K-means Clustering Results",
       x = "Age",
       y = "Cholesterol") +
  theme_minimal()

