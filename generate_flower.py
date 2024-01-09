# Author      : Madhumitha Sukumar
# Time period : 3 Jan 2023 - 
# Description : 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console

# PART 1 

# Load the iris Dataset
file_path = "Iris.csv"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(file_path, names =column_names)

def printline():
    print("---------------------------------------------------------")
   

# Print the title and the number of rows and columns
rows , columns = iris_data.shape
console = Console() # Create a Console object
console.print("\t\t\t[bold red]FLOWER PREDICTION USING IRIS DATA[/bold red]\n")
printline()
print("\nData Analysis on Iris dataset \n")
print(f"Total Rows : {rows} \tColumns : {columns}")

# Data cleaning
# 1. Missing values
print("\nMissing values ",iris_data.isnull().sum())
print()

# 2. Duplicate Values
duplicates = iris_data.duplicated()
print("Number of duplicate rows:", duplicates.sum(),"\n")
printline()
iris_data = iris_data.drop_duplicates()#remove if any

#PART 2

# Prepare the features (X) and target variable (y)
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier 
model = DecisionTreeClassifier(random_state=42) # using same random seed = 42

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model performance
console.print("\n[yellow]Model Performance[yellow]\n")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Display the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall,"\n")
printline()

# Display detailed classification report
console.print("\n[yellow]Classification Report:[yellow]\n")
print(classification_report(y_test, y_pred))
printline()