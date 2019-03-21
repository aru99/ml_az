"""
author: Arman
Date : 21/3/19
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:, 2].values

# Fitting random forest regression to the dataset
from sklearn.ensemble import RandomForestRegressor
# n_estimators = number of trees
regressor = RandomForestRegressor(n_estimators=300, criterion='mse', random_state=0)
regressor.fit(X, Y)
# predicting the new result
Y_pred = regressor.predict([[6.5]])

# visualising the regression result
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y, color='red', marker='x')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Random forest regression")
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()













