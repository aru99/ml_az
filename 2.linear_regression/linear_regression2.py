import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:200, :-1].values
Y = dataset.iloc[:200, 1]

# splitting the data_set into traning set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

# fitting simple linear regression to the traning set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# predicting the test set result
Y_pred = regressor.predict(X_test)

# Visualising the Traning set
plt.scatter(X_train, Y_train, color='red', marker='x')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("number V/S predict (Traning set)")
plt.xlabel("number")
plt.ylabel("predict")
plt.show()

# Visualising the Traning set
plt.scatter(X_test, Y_test, color='red', marker='x')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("number V/S predict (Test set)")
plt.xlabel("number")
plt.ylabel("predict")
plt.show()
