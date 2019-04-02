# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# immporting another library for traning set and test set
from sklearn.model_selection import train_test_split
# importing library for feature scaling
from sklearn.preprocessing import StandardScaler
# importing library for logisticRegression
from sklearn.linear_model import LogisticRegression
# importing library for confusion matrix
from sklearn.metrics import confusion_matrix
# importing library for visualisation
from matplotlib.colors import ListedColormap

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

# fitting the logistic regression to the data set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

# predicting the results
Y_pred = classifier.predict(X_test)

# making the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)




