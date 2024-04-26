import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Data Preprocessing
data = pd.read_csv('drug.csv')

# Drop rows with missing values
data.dropna(inplace=True)

# Encoding categorical variables
data_encoded = pd.get_dummies(data, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)

# Display the preprocessed data
print(data_encoded.head())

# Repeat the experiment five times with different random splits
for i in range(5):
    # Split the data into features (X) and target variable (y)
    X = data_encoded.drop('Drug', axis=1)
    y = data_encoded['Drug']

    # Split the data into training and testing sets (Fixed 70-30 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # Train a decision tree model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate accuracy and display the results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Experiment {i+1} - Accuracy: {accuracy:.4f}, Tree Size: {model.tree_.node_count}")

# Compare the results and select the best model based on accuracy
# The model with the highest overall performance can be selected.

# Initialize lists to store statistics
train_sizes = np.arange(0.3, 0.8, 0.1)
mean_accuracies = []
max_accuracies = []
min_accuracies = []
mean_tree_sizes = []
max_tree_sizes = []
min_tree_sizes = []

# Run the experiment for each training set size
for size in train_sizes:
    size_accuracies = []
    size_tree_sizes = []
    for seed in range(5):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - size, random_state=seed)

        # Train a decision tree model
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = model.predict(X_test)

        # Calculate accuracy and tree size
        accuracy = accuracy_score(y_test, y_pred)
        size_accuracies.append(accuracy)
        size_tree_sizes.append(model.tree_.node_count)

    # Store statistics for each training set size
    mean_accuracies.append(np.mean(size_accuracies))
    max_accuracies.append(np.max(size_accuracies))
    min_accuracies.append(np.min(size_accuracies))
    mean_tree_sizes.append(np.mean(size_tree_sizes))
    max_tree_sizes.append(np.max(size_tree_sizes))
    min_tree_sizes.append(np.min(size_tree_sizes))

# Display the statistics
statistics = pd.DataFrame({
    'Training Size': train_sizes,
    'Mean Accuracy': mean_accuracies,
    'Max Accuracy': max_accuracies,
    'Min Accuracy': min_accuracies,
    'Mean Tree Size': mean_tree_sizes,
    'Max Tree Size': max_tree_sizes,
    'Min Tree Size': min_tree_sizes
})

print(statistics)

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_sizes, mean_accuracies, label='Mean Accuracy')
plt.plot(train_sizes, max_accuracies, label='Max Accuracy')
plt.plot(train_sizes, min_accuracies, label='Min Accuracy')
plt.title('Accuracy vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_sizes, mean_tree_sizes, label='Mean Tree Size')
plt.plot(train_sizes, max_tree_sizes, label='Max Tree Size')
plt.plot(train_sizes, min_tree_sizes, label='Min Tree Size')
plt.title('Tree Size vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Tree Size')
plt.legend()

plt.tight_layout()
plt.show()