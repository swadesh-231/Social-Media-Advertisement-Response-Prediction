# Load required packages
library(caTools)
library(e1071)
library(ElemStatLearn)

# Read the dataset
dataset <- read.csv('social.csv')

# Subset the dataset columns
dataset <- dataset[, 3:5]

# Encode the target feature as a factor
dataset$Purchased <- factor(dataset$Purchased, levels = c(0, 1))

# Split the dataset into training and test sets
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Scale the features
training_set[, -3] <- scale(training_set[, -3])
test_set[, -3] <- scale(test_set[, -3])

# Train the SVM classifier
classifier <- svm(Purchased ~ ., data = training_set, type = 'C-classification', kernel = 'linear')

# Predict on the test set
y_pred <- predict(classifier, newdata = test_set[-3])

# Create the confusion matrix
cm <- table(test_set[, 3], y_pred)
cm

# Plot the training set results
X1 <- seq(min(training_set[, 1]) - 1, max(training_set[, 1]) + 1, by = 0.01)
X2 <- seq(min(training_set[, 2]) - 1, max(training_set[, 2]) + 1, by = 0.01)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) <- c('Age', 'EstimatedSalary')
y_grid <- predict(classifier, newdata = grid_set)

plot(training_set[, -3], main = 'SVM (Training Set)', xlab = 'Age', ylab = 'Estimated Salary', xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'coral1', 'aquamarine'))
points(training_set, pch = 21, bg = ifelse(training_set[, 3] == 1, 'green4', 'red3'))

# Plot the test set results
X1 <- seq(min(test_set[, 1]) - 1, max(test_set[, 1]) + 1, by = 0.01)
X2 <- seq(min(test_set[, 2]) - 1, max(test_set[, 2]) + 1, by = 0.01)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) <- c('Age', 'EstimatedSalary')
y_grid <- predict(classifier, newdata = grid_set)

plot(test_set[, -3], main = 'SVM (Test Set)', xlab = 'Age', ylab = 'Estimated Salary', xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'coral1', 'aquamarine'))
points(test_set, pch = 21, bg = ifelse(test_set[, 3] == 1, 'green4', 'red3'))