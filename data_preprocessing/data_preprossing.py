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