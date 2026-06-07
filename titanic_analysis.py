# Titanic Dataset Analysis
# Import Required Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------
# Load Dataset
# ------------------------------------

df = pd.read_csv("titanic.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------
# Data Cleaning
# ------------------------------------

# Fill missing Age values with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked values with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column because it has too many missing values
df.drop('Cabin', axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ------------------------------------
# Question 1
# Who survived more: males or females?
# ------------------------------------

gender_survival = df.groupby('Sex')['Survived'].mean()

print("\nSurvival Rate by Gender:")
print(gender_survival)

# ------------------------------------
# Question 2
# Did passenger class affect survival?
# ------------------------------------

class_survival = df.groupby('Pclass')['Survived'].mean()

print("\nSurvival Rate by Passenger Class:")
print(class_survival)

# ------------------------------------
# Question 3
# What was the survival rate by age group?
# ------------------------------------

bins = [0, 12, 18, 35, 60, 100]
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']

df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

age_survival = df.groupby('AgeGroup')['Survived'].mean()

print("\nSurvival Rate by Age Group:")
print(age_survival)

# ------------------------------------
# Visualization 1
# Bar Chart of Survival by Gender
# ------------------------------------

plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)

plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')

plt.show()

# ------------------------------------
# Visualization 2
# Bar Chart of Survival by Class
# ------------------------------------

plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df)

plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

plt.show()

# ------------------------------------
# Visualization 3
# Histogram of Passenger Ages
# ------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)

plt.title('Passenger Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')

plt.show()

# ------------------------------------
# Visualization 4
# Survival Rate by Age Group
# ------------------------------------

plt.figure(figsize=(8,5))
sns.barplot(x=age_survival.index, y=age_survival.values)

plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')

plt.show()

# ------------------------------------
# Conclusions
# ------------------------------------

print("\nCONCLUSIONS:")
print("1. Females had a higher survival rate than males.")
print("2. First-class passengers had better survival chances.")
print("3. Age had an impact on survival rates.")
print("4. Missing values were cleaned successfully.")
print("5. Visualizations helped identify important survival patterns.")