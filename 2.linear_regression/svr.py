
"""
author : Mohammad Arman
date : 15/3/19
"""
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
# for feature scaling
from sklearn.preprocessing import StandardScaler
# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# feature scaling
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)


# fitting svr to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, Y)

# predicting new result
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
print(Y_pred)

# visualising SVR result
plt.scatter(X,Y, color='red',marker='x')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('SVR')
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()
























#