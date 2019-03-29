# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# immporting another library for traning set and test set
from sklearn.model_selection import train_test_split

# importing the data set 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

# splitting the dataset into traning set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


