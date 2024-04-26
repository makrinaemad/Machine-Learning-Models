import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data
df = pd.read_csv("loan_old.csv")
missingValues = df.isnull().sum()
missingValues

# check type
dataTypes = df.dtypes
dataTypes


numerical = df.select_dtypes(include=['float64', 'int64'])
scaleInfo = numerical.describe()
scaleInfo

plt.figure(figsize=(10, 6))
sns.boxplot(data=numerical)
plt.show()

sns.pairplot(numerical)
plt.show()

# Drop
df.dropna(inplace=True)

# features and targets
features = df.drop(['Loan_ID','Max_Loan_Amount', 'Loan_Status'], axis=1)
target1 = df['Max_Loan_Amount']
target2 = df['Loan_Status']

X_train, X_test, y_train1, y_test1, y_train2, y_test2 = train_test_split(features, target1, target2,test_size=0.2, random_state=0,shuffle=True)

# Encode
categorical_features = X_train.select_dtypes(include=['object']).columns
X_train[categorical_features] = X_train[categorical_features].apply(LabelEncoder().fit_transform)
X_test[categorical_features] = X_test[categorical_features].apply(LabelEncoder().fit_transform)
label_encoder = LabelEncoder()
y_train2 = label_encoder.fit_transform(y_train2)
y_test2 = label_encoder.fit_transform(y_test2)

numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.fit_transform(X_test[numerical_features])


print("X_train:")
print(X_train.head())
print("\nX_test:")
print(X_test.head())
print("\ny_train1:")
print(y_train1)
print("\ny_test1:")
print(y_test1)
print("\ny_train2:")
print(y_train2)
print("\ny_test2:")
print(y_test2)

#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train1)
y_pred = model.predict(X_test)
r2 = r2_score(y_test1, y_pred)
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train1, y_train_pred)
print(f'R-squared Score on Training Data: {r2_train}')

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test1, y_pred)
y_pred = model.predict(X_test)
r2 = r2_score(y_test1, y_pred)
print(f'R-squared Score on Test Data:  {r2}')


#logistic regression
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def costFUN(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = -1 / m * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    costs = []

    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = X.T @ (h - y) / m
        theta -= learning_rate * gradient
        cost = costFUN(X, y, theta)
        costs.append(cost)

    return theta, costs
def predict(X, theta):
    return np.round(sigmoid(X @ theta))

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train2_np = y_train2.reshape(-1, 1)
y_test2_np = y_test2.reshape(-1, 1)

X_train_np = np.hstack((np.ones((X_train_np.shape[0], 1)), X_train_np))
X_test_np = np.hstack((np.ones((X_test_np.shape[0], 1)), X_test_np))

theta_initial = np.zeros((X_train_np.shape[1], 1))

learning_rate = 0.001
iterations = 1000
theta_trained, costs = gradient_descent(X_train_np, y_train2_np, theta_initial, learning_rate, iterations)

y_pred_logreg = predict(X_test_np, theta_trained)

def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy

accuracy = accuracy(y_test2_np, y_pred_logreg)
print(f'Accuracy of the logistic regression model: {accuracy}')


# Load the dataset
df2 = pd.read_csv("loan_new.csv")
missingValues1 = df2.isnull().sum()
missingValues1

# Drop
df2.dropna(inplace=True)
df2.drop('Loan_ID',axis=1,inplace=True)

# Encode
categorical_features1 = df2.select_dtypes(include=['object']).columns
df2[categorical_features1] = df2[categorical_features1].apply(LabelEncoder().fit_transform)
print(df2)

numerical_features2 = df2.select_dtypes(include=['float64', 'int64']).columns
df2[numerical_features] = scaler.fit_transform(df2[numerical_features])
print(df2)

X_test_new = df2

# linear regression
y_pred_linear = model.predict(X_test_new)
print("Linear Regression Predictions:")
print(y_pred_linear)

#predict logistic regression
X_test_new_np = X_test_new.to_numpy()
X_test_new_np = np.hstack((np.ones((X_test_new_np.shape[0], 1)), X_test_new_np))
y_pred_logistic = predict(X_test_new_np, theta_trained)
print("Logistic Regression Predictions:")
print(y_pred_logistic)

