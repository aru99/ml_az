
"""
author : Mohammad Arman
date : 15/3/19
"""
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# fitting svr to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, Y)

# predicting new result
Y_pred = regressor.predict([[6.5]])
print(Y_pred)

# visualising SVR result
# plt.scatter(X,Y, color='red',marker='x')
# plt.plot(X, regressor.predict(X), color='blue')


























#