import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset (tab-separated)
df = pd.read_csv(r"G:\My Drive\taitanicdataset.csv")
# Get a summary of the data
print(df.info())

# Get descriptive statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check the distribution of some key variables

# For example, the distribution of age
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# The distribution of fare
sns.histplot(df['Fare'], kde=True)
plt.title('Fare Distribution')
plt.show()

sns.countplot(x='Sex', data=df)
plt.title('Passenger Gender Count')
plt.show()

sns.countplot(x='Pclass', data=df)
plt.title('Passenger Class Count')
plt.show()

sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age vs Survival')
plt.show()

sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare vs Survival')
plt.show()

sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Age vs Fare by Survival')
plt.show()

# Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# Correlation heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Numeric Columns Only)')
plt.show()
