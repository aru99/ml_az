"""
author : Mohammad Arman
Date : 13/3/19
"""
# multiple linear regression

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
# importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# encoding categorical data
# encoding the independent variable
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# fitting the multiple linear regression to the training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# predicting the test results
Y_pred = regressor.predict(X_test)

# building the optimal model using the Backward Elimination
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# creating a matrix of all the independent variables
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# calling the ols class
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

# without dummy variable 2
# creating a matrix of all the independent variables
X_opt = X[:, [0, 1, 3, 4, 5]]
# calling the ols class
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

# without dummy variable 1
# creating a matrix of all the independent variables
X_opt = X[:, [0, 3, 4, 5]]
# calling the ols class
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())


# without 4
# creating a matrix of all the independent variables
X_opt = X[:, [0, 3, 5]]
# calling the ols class
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

# final OLS model to be used.
# creating a matrix of all the independent variables
X_opt = X[:, [0, 3]]
# calling the ols class
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())



#plots
plt.scatter()
plt.xlabel('')
plt.ylabel('')



