"""
Author : Mohammad Arman
Date : 14/3/19
Subject : polynomial regression
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
# for X, making sure that it is a matrix not a vector, thus X = dataset.iloc[:, 1:2].values
# insted of X = dataset.iloc[:,1].values
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, -1].values
# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# polynomial regression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)
