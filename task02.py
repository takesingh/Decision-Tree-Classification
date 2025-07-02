# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

df = pd.read_csv("train.csv")  


print("Initial Dataset:")
print(df.head())


print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())


df['Age'].fillna(df['Age'].median(), inplace=True)


df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


df.drop(columns=['Cabin'], inplace=True)

df.dropna(inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()


plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Between Features')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age Distribution Across Passenger Classes')
plt.show()


print("\nSummary Statistics:")
print(df.describe(include='all'))


df.to_csv("cleaned_titanic.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_titanic.csv'")
