"""
Author : Arman
Date : 2/4/19

implementation of K nearest neighbour classification algorithm
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# library for splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
# library for feature scaling
from sklearn.preprocessing import StandardScaler


# importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

# training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


