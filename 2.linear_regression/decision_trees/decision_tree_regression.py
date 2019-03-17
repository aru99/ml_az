# Decision Tree Regression

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# matrix of features
X = dataset.iloc[:, 1:2].values
# dependant variable vector
Y = dataset.iloc[:, 2].values

# fitting the Decision tree to the dataset
regressor = DecisionTreeRegressor(criterion="mse", random_state=0)
regressor.fit(X, Y)

# predicting a new result
Y_pred = regressor.predict([[6.5]])

# visualising the result (low res)
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X), color="blue")
plt.title("regression model plot 1 (Decision tree)")
plt.xlabel('Position')
plt.ylabel("slalry")
plt.show()

# visualising the result (high res)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid,regressor.predict(X_grid), color='blue')
plt.title('regressor model (decision tree high res)')
plt.xlabel('position')
plt.ylabel('slalary')
plt.show()

#another visualisation 
plt.scatter(X,Y,color='red')
plt.scatter(X,regressor.predict(X),color = 'blue')
plt.show()
