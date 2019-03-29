# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# immporting another library for traning set and test set
from sklearn.model_selection import train_test_split
# importing library for feature scaling
from sklearn.preprocessing import StandardScaler

# importing the data set 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

# splitting the dataset into traning set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
