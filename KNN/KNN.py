import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("diabetes.csv")

# Splitting the data into features (x) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Splitting the dataset into training (70%) and testing (30%) sets
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=0)

#Min-Max Scaling Function
def minMaxScaling(XTrain, XTest):
    XTrainScaled = np.copy(XTrain)
    XTestScaled = np.copy(XTest)
    
    for i in range(XTrain.shape[1]):  
        minVal = np.min(XTrain[:, i])
        maxVal = np.max(XTrain[:, i])
        
        XTrainScaled[:, i] = (XTrain[:, i] - minVal) / (maxVal - minVal)
        
        XTestScaled[:, i] = (XTest[:, i] - minVal) / (maxVal - minVal)

        
        
    return XTrainScaled,XTestScaled

#Sonvert DataFrame to numpy arrays for scaling
XTrain_np = XTrain.to_numpy()
XTest_np = XTest.to_numpy()


yTrain_np = yTrain.to_numpy()
yTest_np = yTest.to_numpy()

XTrainScaled, XTestScaled =minMaxScaling(XTrain_np, XTest_np)

# KNN Implementation
def euclideanDistance(instance1, instance2):
    return np.sqrt(np.sum((instance1 - instance2) ** 2))

def knnPredict(XTrain, yTrain, XTestInstance, k):
    distances = []
    for i in range(len(XTrain)):
        dist = euclideanDistance(XTrain[i], XTestInstance)
        distances.append((yTrain[i], dist))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]

    weights = {0: 0, 1: 0}
    for neighbor in neighbors:
        label, dist = neighbor
        weight = 1 / (dist + 0.001)  
        weights[label] += weight

    return max(weights, key=weights.get)

def knn(XTrain, yTrain, XTest, yTest, kValues):
    accuracies = []
    for k in kValues:
        correctPredictions = 0
        for i in range(len(XTest)):
            predictedClass = knnPredict(XTrain, yTrain, XTest[i], k)
            if predictedClass == yTest[i]:
                correctPredictions += 1
        accuracy = correctPredictions / len(XTest) * 100
        accuracies.append((k, correctPredictions, len(XTest), accuracy))
    return accuracies

kValues = [1, 3, 5, 7, 9]


knnAccuracies = knn(XTrainScaled, yTrain_np, XTestScaled, yTest_np, kValues)

averageAccuracy = sum([acc[3] for acc in knnAccuracies]) / len(knnAccuracies)

for k, correct, total, accuracy in knnAccuracies:
    print(f"k value: {k}")
    print(f"Number of correctly classified instances: {correct}")
    print(f"Total number of instances in the test set: {total}")
    print(f"Accuracy: {accuracy:.2f}%\n")

print(f"Average Accuracy Across All Iterations: {averageAccuracy:.2f}%")







