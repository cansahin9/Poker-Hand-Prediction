#-----Section 01: Setup and Libraries-------------------------------------------
# set working directory and load data

setwd(dirname(file.choose()))

# Print the current working directory to confirm the change
getwd()

# Load necessary libraries for data handling and visualization
library(readr)
library(dplyr)
library(ggplot2)
library(nnet)
library(caret)
library(randomForest)
library(corrplot)
library(psych)
if (!require("visdat")) install.packages("visdat")
library(visdat)
if (!require("DiagrammeR")) install.packages("DiagrammeR")
library(DiagrammeR)

#-----Section 02: Data Loading and Preparation----------------------------------

# Load the training and testing datasets
training_data <- read_csv("poker-hand-training-true.data")
testing_data <- read_csv("poker-hand-testing.data")

# Assign meaningful column names to the datasets
col_names <- c('S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Class')
colnames(training_data) <- col_names
colnames(testing_data) <- col_names

# Convert the 'Class' column to a factor
training_data$Class <- factor(training_data$Class)
testing_data$Class <- factor(testing_data$Class, levels = levels(training_data$Class))

# Define the columns for rank and suit for correlation analysis
rank_columns <- c("C1", "C2", "C3", "C4", "C5")
suits_columns <- c("S1", "S2", "S3", "S4", "S5")

#-----Section 03: Data Exploration---------------------------------------------

# Display basic summary statistics for training and testing data
summary(training_data)
summary(testing_data)

# Visualize the distribution of poker hand classes
ggplot(training_data, aes(x = factor(Class))) +
  geom_bar() +
  labs(title = "Distribution of Poker Hand Classes", x = "Poker Hand Class", y = "Frequency")

# Generate pairwise plots to explore data relationships
pairs(~S1 + C1 + S2 + C2 + S3 + C3 + S4 + C4 + S5 + C5 + Class, data = training_data)

#-----Section 04: Distribution Analysis-----------------------------------------

# Plot histograms for suit distributions
par(mfrow = c(1, 5))
for (i in seq(1, 9, by = 2)) {
  hist(training_data[[i]], main = paste("Distribution of Suit", i), xlab = "Suit", breaks = 4)
}

# Plot histograms for rank distributions
par(mfrow = c(1, 5))
for (i in seq(2, 10, by = 2)) {
  hist(training_data[[i]], main = paste("Distribution of Rank", i - 1), xlab = "Rank", breaks = 13)
}

#-----Section 05: Correlation Analysis------------------------------------------
# Calculate and print correlation matrices for ranks and suits
cor_rank <- cor(training_data[rank_columns])
cor_suits <- cor(training_data[suits_columns])

# Visualize correlation with heatmaps
corrplot(cor_rank, method = "shade", tl.col = "black", tl.srt = 45)
corrplot(cor_suits, method = "shade", tl.col = "black", tl.srt = 45)


#-----Section 06: Data Quality Checks-------------------------------------------

# Check for missing values
sum(is.na(training_data))

vis_miss(training_data)

#-----Section 07: Additional Data Visualization---------------------------------

# Create a boxplot for the first card's rank across different poker hand classes
ggplot(training_data, aes(x = factor(Class), y = C1)) +
  geom_boxplot() +
  labs(title = "Boxplot of Card Ranks by Poker Hand Class", x = "Poker Hand Class", y = "Rank of First Card") +
  theme_minimal()

#-----Section 08: Hyperparameter Tuning and Cross-Validation-------------------

# Setup training control parameters for cross-validation
fitControl <- trainControl(
  method = "cv",  # Cross-validation
  number = 10,    # Number of folds
  verboseIter = TRUE  # Print training iterations
)

# Define a grid of hyperparameters to tune
tuneGrid <- expand.grid(
  .size = c(20, 30),     # Number of neurons in the hidden layer
  .decay = c(0.01, 0.1)  # Weight decay for regularization
)

# Train the model using caret's train function with nnet
nn_model_cv <- train(
  Class ~ ., 
  data = training_data,
  method = "nnet", 
  trControl = fitControl,
  tuneGrid = tuneGrid,
  maxit = 500, 
  linout = FALSE, 
  trace = FALSE
)

# Output the best model's results and parameters
print(nn_model_cv)
summary(nn_model_cv)

#-----Section 09: Neural Network Modeling---------------------------------------

# Set seed for reproducibility
set.seed(123)

# Neural network setup and training using optimized parameters from cross-validation
best_size <- nn_model_cv$bestTune$size
best_decay <- nn_model_cv$bestTune$decay

nn_model <- nnet(Class ~ ., data = training_data, size = best_size, decay = best_decay, maxit = 500, linout=FALSE, trace = FALSE)
print(nn_model)  # Print the model summary

# Predict and evaluate the neural network model
nn_predictions <- predict(nn_model, testing_data, type = "class")
nn_confusionMatrix <- table(Predicted = nn_predictions, Actual = testing_data$Class)
print(nn_confusionMatrix)

# Calculate and print accuracy
nn_accuracy <- sum(diag(nn_confusionMatrix)) / sum(nn_confusionMatrix)
print(paste("NN Model Accuracy:", nn_accuracy))


#-----Section 10: Model Comparison and Visualization----------------------------

# Comparing accuracies of ANN and Random Forest models
ann_accuracy <- sum(diag(nn_confusionMatrix)) / sum(nn_confusionMatrix)
rf_accuracy <- sum(diag(rf_confusionMatrix)) / sum(rf_confusionMatrix)

# Print and compare accuracies
print(paste("ANN Accuracy:", ann_accuracy))
print(paste("Random Forest Accuracy:", rf_accuracy))

# Combine the accuracies into a vector and convert to a data frame for plotting
accuracies <- c(ANN = ann_accuracy, Random_Forest = rf_accuracy)
accuracy_df <- data.frame(Model = names(accuracies), Accuracy = unname(accuracies))

# Plot the accuracies as a bar chart for visual comparison
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  ylim(0, 1) +
  labs(title = "Model Accuracy Comparison", y = "Accuracy", x = "") +
  theme_minimal()

#-----Section 11: ROC Curve and AUC Calculation for Multi-Class Classification-----------------

library(pROC)  # Load the pROC library for ROC analysis

# Function to calculate and plot ROC for each class
calculate_plot_roc <- function(predictions, data, title_prefix) {
  # Loop through each class, treating it as 'positive' and all others as 'negative'
  for (i in seq_along(levels(data$Class))) {
    # Define the current class as 'positive' and all others as 'negative'
    current_class <- levels(data$Class)[i]
    binary_response <- ifelse(data$Class == current_class, "positive", "negative")
    
    # Calculate ROC for the current class
    roc_result <- roc(binary_response, predictions[, i])
    
    # Print AUC
    cat(sprintf("AUC for %s - class %s: %0.2f\n", title_prefix, current_class, auc(roc_result)))
    
    # Plot ROC
    plot(roc_result, main = paste(title_prefix, "ROC Curve for", current_class))
  }
}
dev.off()
# Calculate and plot ROC for Neural Network Model
calculate_plot_roc(nn_probabilities, testing_data, "Neural Network")

# Calculate and plot ROC for Random Forest Model
calculate_plot_roc(rf_probabilities, testing_data, "Random Forest")

# Create a more detailed and structured graph using DiagrammeR's grViz function
nn_diagram <- grViz("
  digraph neural_network {
    graph [layout = dot, rankdir = TB, nodesep = 1, ranksep = 2]
    
    # Define node styles
    node [shape = circle, style = filled, fillcolor = lightblue2, fontcolor = black, fontsize = 12, fontname = Helvetica]
    edge [color = gray40, penwidth = 2]
    
    # Define subgraph clusters for better visual organization
    subgraph cluster_input {
      label = 'Input Layer'
      color = transparent
      input1 [label = 'Input 1', fillcolor = lightcoral]
      input2 [label = 'Input 2', fillcolor = lightcoral]
      input3 [label = 'Input 3', fillcolor = lightcoral]
    }

    subgraph cluster_hidden1 {
      label = 'Hidden Layer 1'
      color = transparent
      hidden11 [label = 'Neuron 1']
      hidden12 [label = 'Neuron 2']
    }

    subgraph cluster_hidden2 {
      label = 'Hidden Layer 2'
      color = transparent
      hidden21 [label = 'Neuron 1']
    }

    subgraph cluster_output {
      label = 'Output Layer'
      color = transparent
      output [label = 'Output', fillcolor = salmon]
    }

    # Connections between nodes
    input1 -> hidden11
    input1 -> hidden12
    input2 -> hidden11
    input2 -> hidden12
    input3 -> hidden11
    input3 -> hidden12
    hidden11 -> hidden21
    hidden12 -> hidden21
    hidden21 -> output
  }
")

# Display the graph
print(nn_diagram)

#-----Section 12: Cleanup-------------------------------------------

# Remove all variables from the environment
rm(list = ls())
