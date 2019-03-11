#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:26:10 2019

@author: aru
#data preprossing
"""

#importing libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#importing dataset 
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values 


#taking care of missing data 
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encoding the categorial data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:,0]=label_encoder_X.fit_transform(X[:,0])
one_hot_encoder = OneHotEncoder(categorical_features = [0])
X = one_hot_encoder.fit_transform(X).toarray()
#labelEncoding the purchased coloumn 
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

#splitting the data set into the traning set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)






















