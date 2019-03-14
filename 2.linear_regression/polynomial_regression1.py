"""
Author : Mohammad Arman
Date : 14/3/19
Subject : polynomial regression
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values
Y = dataset.iloc[:, -1].values



