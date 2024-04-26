Project Title: Machine Learning Projects
This repository contains the code for four machine learning tasks. Each task addresses a specific problem statement and involves implementing various machine learning algorithms and techniques.

Task 1: MNIST Digit Recognition
Objective:
Acquire proficiency in using TensorFlow and Scikit-Learn.
Familiarize with Artificial Neural Networks (ANNs) and K-Nearest Neighbors (K-NN).
Understand the concept of hyperparameter tuning.
Approach:
Data Exploration and Preparation:
Load the MNIST dataset from CSV files.
Perform initial data exploration.
Normalize pixel values and resize images.
Split the training data into training and validation sets.
Experiments and Results:
Implement K-NN algorithm for classification.
Experiment with different architectures of ANN.
Compare performance and select the best model.
Evaluate model using confusion matrix and test data.
Task 2: Loan Eligibility Prediction
Objective:
Build linear and logistic regression models to predict loan decisions and amounts based on applicant details.
Approach:
Data Analysis:
Load and analyze "loan_old.csv" dataset.
Handle missing values, feature types, and scale.
Visualize relationships between numerical features.
Data Preprocessing:
Remove records with missing values.
Split data into features and targets.
Encode categorical features and targets.
Standardize numerical features.
Model Building:
Fit linear regression model using sklearn.
Evaluate model using R2 score.
Implement logistic regression from scratch using gradient descent.
Calculate accuracy of logistic regression model.
Prediction:
Load and preprocess "loan_new.csv" dataset.
Predict loan amounts and status using trained models.
Task 3: Medication Prediction using Decision Trees
Objective:
Predict appropriate medication for patients based on their attributes using decision trees.
Approach:
Data Preprocessing:
Handle missing data and encode categorical variables.
Experiment 1:
Train and test decision trees with fixed train-test split ratio.
Report accuracies of different models.
Experiment 2:
Vary train-test split ratio from 30% to 70%.
Calculate statistics and plot accuracy against training set size.
Analyze tree size against training set size.
Task 4: Diabetes Classification using K-Nearest Neighbors
Objective:
Implement a KNN classifier to classify diabetes instances.
Approach:
Data Preprocessing:
Normalize feature columns using Log Transformation or Min-Max Scaling.
KNN Implementation:
Implement KNN classifier without using built-in functions.
Divide data into training and testing sets.
Break Ties using Distance-Weighted Voting:
Assign higher weights to closer neighbors to break ties.
Output:
Output the value of k and summary information for each iteration.
Calculate and output average accuracy across all iterations.
