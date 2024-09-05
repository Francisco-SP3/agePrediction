# Francisco Salas Porras A01177893
# Age Prediction using Random Forest

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
import csv

###################################### Hyperparameters ######################################
# Learning rate
alpha = 0.01


###################################### Functions ######################################



###################################### Clean dataset ######################################

# Load the data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# Show the data
print("----------------------------------------------------------------")
print("Train data\n")
print(train.shape)
print(train.head())
print("----------------------------------------------------------------")
print("Test data\n")
print(test.shape)
print(test.head())

# Check data types
print("----------------------------------------------------------------")
print("Data types\n")
print(train.dtypes)

# Check for missing values
print("----------------------------------------------------------------")
print("Train missing values\n")
print(train.isnull().sum())
print("----------------------------------------------------------------")
print("Test missing values\n")
print(test.isnull().sum())

# Check for repeated values
print("----------------------------------------------------------------")
print("Train repeated values")
print(train.duplicated().sum())
print("----------------------------------------------------------------")
print("Test repeated values")
print(test.duplicated().sum())

# Train and test data seem to be the exact same, so we will use the train data

###################################### Training ######################################


# Split label and features
#X_train = train.drop('Age', axis=1)
#y_train = train['Age']