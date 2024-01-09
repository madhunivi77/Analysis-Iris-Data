# Author      : Madhumitha Sukumar
# Time period : 3 Jan 2023 - 9 Jan 2023
# Description : This Python script performs data cleaning, analyzes the Iris dataset 
#               through visualizations, and implements a Decision Tree classifier, 
#               providing key insights and model evaluation metrics
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
console.print("\n\t\t\t[bold red]FLOWER PREDICTION USING IRIS DATA[/bold red]\n")
print("\nData Analysis on Iris dataset \n")
print(f"Total Rows : {rows} \tColumns : {columns}")
#print(iris_data.head()) -- top 5 rows
#print(df.info()) -- column data types

# Data cleaning

# 1. Missing values
print("\nMissing values\n",iris_data.isnull().sum())
print()

# 2. Duplicate Values
data = iris_data.drop_duplicates(subset ="species",)
data

# 3. Balanced data set
# species contain equal amounts of rows or not
print(iris_data.value_counts("species")) 
printline()

#PART 2 - Data Science Task ( ML Algorithm - Decision trees)

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

# PART 3 - Exploratory Data Analysis (EDA)

# Display a countplot of the Species
sns.countplot(x='species', data= iris_data, )
plt.show()

# Creating histograms for each category
iris_data.hist(bins=20, figsize=(10, 7))
plt.suptitle("Distribution of Features in Iris Dataset", y=0.99, fontsize=16)
plt.show()

# Scatterplot for Sepal
sns.scatterplot(x='sepal_length', y='sepal_width',
                hue='species', data=iris_data, )
plt.legend(bbox_to_anchor=(0.67,0.99), loc=2)
plt.show()

# Scatterplot for Petal
sns.scatterplot(x='petal_length', y='petal_width',
                hue='species', data=iris_data, )
plt.legend(bbox_to_anchor=(0.67,0.3), loc=2)
plt.show()

# Box plots to visualize the distribution of each feature by species
plt.figure(figsize=(10, 7))
for i, feature in enumerate(column_names[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=feature, data=iris_data ,palette='Set2')
    plt.title(f'{feature} distribution by species')
plt.suptitle("Box Plots of Features by Species", y=0.99, fontsize=16)
plt.tight_layout()
plt.show()