import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('titanic.csv')
df.info()

df.head()

df.tail()

df.isnull().sum()

df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)

df.dropna(subset=['Survived'], inplace=True)

df.isnull().sum()

duplicates = df[df.duplicated()]
print("Duplicate rows:\n", duplicates)

sns.boxplot(df)
plt.show()

Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['Age'] = df['Age'].clip(lower=lower_bound, upper=upper_bound)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])
print("Data after encoding:\n", df.head())

# Univariate Analysis
print("Univariate Analysis of Age:\n", df['Age'].describe())
# Bivariate Analysis
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.show()
# Multivariate Analysis
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
plt.show()

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
print("Data after scaling:\n", df.head())

X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,␣
↪random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
