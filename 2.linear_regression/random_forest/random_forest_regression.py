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
regressor = RandomForestRegressor(n_estimators=10, criterion='mse', random_state=0)
regressor.fit(X, Y)
# pre














