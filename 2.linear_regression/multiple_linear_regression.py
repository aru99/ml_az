"""
author : Mohammad Arman
Date : 13/3/19
"""
# multiple linear regression

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

#encoding categorical data


