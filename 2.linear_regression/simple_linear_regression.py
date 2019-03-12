#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 04:44:15 2019

@author: aru
"""
#simple linear regression 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data set 
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,1].values 

#splitting the data set into the traning set and test set 
from sklearn.cross_validation import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1/3, random_state =0)

#fitting simple linear regression to the trainig set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)