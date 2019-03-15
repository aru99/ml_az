"""
Author : Arman
Date : 15/3/19
"""
# regression Template


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv("")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# splitting the dataset into traning set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# featuring scaling
"""
do if required
"""

# Fitting the regression Model to the database
# create your regressor


# predicting the result with regression model
Y_pred = regressor.predict(6.5)

# visualising the regression results
plt.scatter(X,Y, color='red')
plt.plot(X, regressor.predict(X), color="blue")
plt.title("regression model result")
plt.xlabel()
plt.ylabel()
plt.show()


# visualising the regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
# comverting the above vector into a matrix
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y, color='red')
plt.plot(X, regressor.predict(X), color="blue")
plt.title("regression model result")
plt.xlabel()
plt.ylabel()
plt.show()
















