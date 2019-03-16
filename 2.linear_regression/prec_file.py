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
print(Y)
Y = Y.reshape(-1,1)
print(Y)
# feature scaling
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)