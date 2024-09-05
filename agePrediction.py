# Francisco Salas Porras A01177893
# Age Prediction using Random Forest

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math
import csv
#from sklearn.model_selection import data_test_split

###################################### Hyperparameters ######################################
# Learning rate
alpha = 0.01


###################################### Functions ######################################



###################################### Clean dataset ######################################

# Load the data
data = pd.read_csv('Train.csv')

# Show the data
print("----------------------------------------------------------------")
print("Dataset\n")
print(data.shape)
print(data.head())

# Check data types
print("----------------------------------------------------------------")
print("Data types\n")
print(data.dtypes)

# Check for missing values
print("----------------------------------------------------------------")
print("Missing values\n")
print(data.isnull().sum())

# Check for repeated values
print("----------------------------------------------------------------")
print("Repeated values")
print(data.duplicated().sum())

###################################### Training ######################################


# Split label and features
#X_data = data.drop('Age (years)', axis=1)
#y_data = data['Age (years)']