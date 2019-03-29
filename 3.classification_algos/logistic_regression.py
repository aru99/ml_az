# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

