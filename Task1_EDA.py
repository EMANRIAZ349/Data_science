# ==========================================
# TASK 1: TITANIC DATASET - COMPLETE EDA
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

print("=" * 50)
print("TASK 1: LIBRARIES IMPORTED")
print("=" * 50)

# Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("\nDataset loaded successfully!")
print("Shape:", df.shape)

# First 5 rows
print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

# Random 5 rows
print("\n" + "=" * 50)
print("RANDOM 5 ROWS")
print("=" * 50)
print(df.sample(5))

# Info
print("\n" + "=" * 50)
print("DATASET INFO")
print("=" * 50)
df.info()

# Shape
print("\n" + "=" * 50)
print("ROWS AND COLUMNS")
print("=" * 50)
print("Number of rows =", df.shape[0])
print("Number of columns =", df.shape[1])

# Data types
print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)

# Non-missing count
print("\n" + "=" * 50)
print("NON-MISSING VALUES")
print("=" * 50)
print(df.notnull().sum())

# Duplicates
print("\n" + "=" * 50)
print("DUPLICATE ROWS")
print("=" * 50)
print("Duplicate Rows:", df.duplicated().sum())

# Unique values
print("\n" + "=" * 50)
print("UNIQUE VALUES IN 'Survived'")
print("=" * 50)
print("Unique values:", df["Survived"].unique())

# Nunique
print("\n" + "=" * 50)
print("UNIQUE VALUES PER COLUMN")
print("=" * 50)
print(df.nunique())

# Describe
print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)
print(df.describe())

# Missing values
print("\n" + "=" * 50)
print("MISSING VALUES COUNT")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("COLUMNS WITH MISSING VALUES")
print("=" * 50)
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))

print("\n" + "=" * 50)
print("MISSING VALUE PERCENTAGE")
print("=" * 50)
missing_percentage = (df.isnull().sum() / len(df) * 100).round(2)
print(pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": missing_percentage
}))

# Remove duplicates
print("\n" + "=" * 50)
print("REMOVE DUPLICATES")
print("=" * 50)
df = df.drop_duplicates().copy()
print("Shape after removing duplicates:", df.shape)

# Fill Age with median
print("\n" + "=" * 50)
print("FILL MISSING AGE WITH MEDIAN")
print("=" * 50)
median_age = df["Age"].median()
print("Median Age:", median_age)
df["Age"] = df["Age"].fillna(df["Age"].median())
print("Missing Age after filling:", df["Age"].isnull().sum())

# Fill Embarked with mode
print("\n" + "=" * 50)
print("FILL MISSING EMBARKED WITH MODE")
print("=" * 50)
mode_embarked = df["Embarked"].mode()[0]
print("Most common Embarked:", mode_embarked)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print("Missing Embarked after filling:", df["Embarked"].isnull().sum())

print("\n" + "=" * 50)
print("REMAINING MISSING VALUES")
print("=" * 50)
print(df.isnull().sum().sort_values(ascending=False))

# Survival rates
print("\n" + "=" * 50)
print("OVERALL SURVIVAL RATE")
print("=" * 50)
survival_rate = df["Survived"].mean() * 100
print(f"Overall Survival Rate: {survival_rate:.2f}%")

print("\n" + "=" * 50)
print("SURVIVAL RATE BY SEX")
print("=" * 50)
print(df.groupby("Sex")["Survived"].mean() * 100)

print("\n" + "=" * 50)
print("SURVIVAL RATE BY PASSENGER CLASS")
print("=" * 50)
print(df.groupby("Pclass")["Survived"].mean() * 100)

# Histograms
print("\n" + "=" * 50)
print("CREATING HISTOGRAMS...")
print("=" * 50)
numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
df[numeric_cols].hist(bins=30, figsize=(10, 6), layout=(2, 2))
plt.tight_layout()
plt.savefig("histograms.png", dpi=100, bbox_inches="tight")
plt.show()

# Boxplots
print("\n" + "=" * 50)
print("CREATING BOXPLOTS...")
print("=" * 50)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(data=df, y="Age", ax=ax[0, 0])
sns.boxplot(data=df, y="Fare", ax=ax[0, 1])
sns.boxplot(data=df, y="SibSp", ax=ax[1, 0])
sns.boxplot(data=df, y="Parch", ax=ax[1, 1])
plt.tight_layout()
plt.savefig("boxplots.png", dpi=100, bbox_inches="tight")
plt.show()

# Correlation
print("\n" + "=" * 50)
print("CORRELATION MATRIX")
print("=" * 50)
selected_cols = ["Age", "SibSp", "Parch", "Pclass", "Fare"]
print(df[selected_cols].corr())

# Heatmap
print("\n" + "=" * 50)
print("CREATING HEATMAP...")
print("=" * 50)
plt.figure(figsize=(8, 6))
sns.heatmap(df[selected_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png", dpi=100, bbox_inches="tight")
plt.show()

# Survival by Sex bar plot
print("\n" + "=" * 50)
print("CREATING SURVIVAL BY SEX BAR PLOT...")
print("=" * 50)
sns.barplot(x="Sex", y="Survived", data=df, hue="Sex", legend=False)
plt.title("Sex vs Survival")
plt.xlabel("Sex")
plt.ylabel("Average Survival Rate")
plt.savefig("survival_by_sex.png", dpi=100, bbox_inches="tight")
plt.show()

# Top 5 fares
print("\n" + "=" * 50)
print("TOP 5 HIGHEST-PAYING PASSENGERS")
print("=" * 50)
topFares = df.sort_values(by="Fare", ascending=False)
top5 = topFares.head(5)
print(top5[["Name", "Sex", "Pclass", "Fare"]])

sns.barplot(x="Fare", y="Name", data=top5, hue="Sex")
plt.title("Five Highest-Paying Passengers")
plt.xlabel("Fare")
plt.ylabel("Passenger Name")
plt.savefig("top5_fares.png", dpi=100, bbox_inches="tight")
plt.show()

print("\n" + "=" * 50)
print("TASK 1 COMPLETE!")
print("=" * 50)