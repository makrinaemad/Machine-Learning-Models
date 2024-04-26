import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras import layers, models
import io

# Load the dataset
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# Data Exploration and preparation
# Identify the number of unique classes
num_classes = train_data['label'].nunique()
print("Number of Classes:", num_classes)

# Identify the number of features
num_features = len(train_data.columns) - 1  # Subtract 1 for the label column
print("Number of Features:", num_features)

# Check for missing values
missing_values = train_data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Drop rows with null values
train_data = train_data.dropna()

# Normalize pixel values
train_data.iloc[:, 1:] = train_data.iloc[:, 1:] / 255.0
test_data.iloc[:, 1:] = test_data.iloc[:, 1:] / 255.0

# Resize images
image_size = (28, 28)
X = train_data.iloc[:, 1:].values.reshape(-1, *image_size, 1)

# Display the reshaped image
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X[i, :, :, 0], cmap='gray')
    plt.title(f'Label: {train_data.iloc[i, 0]}')
    plt.axis('off')

plt.show()

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, train_data['label'], test_size=0.2, random_state=42)
# K-NN with grid search
# Flatten the images for KNN
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Initialize KNN model
knn = KNeighborsClassifier()

# Define hyperparameter grid for grid search
param_grid_knn = {'n_neighbors': [3, 5, 7]}

# Perform grid search for K-NN
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=3)
grid_search_knn.fit(X_train_flat, y_train)
best_knn = grid_search_knn.best_estimator_

# Train the best K-NN model
best_knn.fit(X_train_flat, y_train)

# Predict on validation set
y_val_pred_knn = best_knn.predict(X_val_flat)

# Calculate accuracy
knn_accuracy = accuracy_score(y_val, y_val_pred_knn)
print("Best K-NN Accuracy: ", knn_accuracy)
# Artificial Neural Network (ANN)
# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# Define and train the second model with different parameters
model2 = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(256, activation='relu'),  # Different hidden layer size
    layers.Dense(num_classes, activation='softmax')
])

# Compile and train the second model with different parameters
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Different learning rate
               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)
# Compare K-NN and ANN
# Compare the outcomes of K-NN and ANN
best_nn_accuracy = max(max(history.history['val_accuracy']), max(history2.history['val_accuracy']))

if knn_accuracy > best_nn_accuracy:
    print("K-NN performed better.")
else:
    print("ANN performed better.")


# Confusion matrix of the best model
conf_matrix = confusion_matrix(y_val, y_val_pred_knn)
print("Confusion Matrix for K-NN:")
print(conf_matrix)

# Save the best model
model.save('best_model.h5')
# Load the best model
loaded_model = tf.keras.models.load_model('best_model.h5')
# Use the best model on the testing data
X_test = test_data.values[:, 1:].reshape(-1, 28, 28, 1) / 255.0  # Normalize and reshape
y_test_pred = np.argmax(loaded_model.predict(X_test), axis=1)

# Save predictions to a CSV file
output_df = pd.DataFrame({'ImageId': range(1, len(y_test_pred) + 1), 'Label': y_test_pred})
output_df.to_csv('predictions.csv', index=False)
# Use the second model on the testing data
X_test = test_data.values[:, 1:].reshape(-1, 28, 28, 1) / 255.0  # Normalize and reshape
y_test_pred2 = np.argmax(model2.predict(X_test), axis=1)
# Optionally, print or use the predictions as needed
print("Predictions for the first model:", y_test_pred)
print("Predictions for the second model:", y_test_pred2)