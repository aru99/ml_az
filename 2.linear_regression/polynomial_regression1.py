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
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# visualising linear regression
plt.scatter(X,Y,color='red',marker="x")
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("truth or bluff (linear Regression)")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

# visualising polynomial regression
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, Y, color="red", marker=".")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title("truth or bluff (ploynomial regression)")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

# both visualisations in the same graph

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, Y, color="red", marker=".")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='red')
plt.plot(X,lin_reg.predict(X),color='green')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title("expected VS predicted ")
plt.xlabel("position level")
plt.ylabel("population")
plt.show()








