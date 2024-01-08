# Author      : Madhumitha Sukumar
# Time period : 3 Jan 2023 - 
# Description : 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris Dataset
file_path = "Iris.csv"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(file_path, names =column_names)

# Data cleaning
# 1. Missing values
print(iris_data.isnull().sum())

# 2. Duplicate Values
duplicates = iris_data.duplicated()
print("Number of duplicate rows:", duplicates.sum())
iris_data = iris_data.drop_duplicates()#remove if any


