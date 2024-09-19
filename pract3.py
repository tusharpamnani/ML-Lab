import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression
df = pd.read_csv('/content/sample_data/50_Startups.csv')

df.head()

df.info()

x = df[['R&D Spend']].values
y = df[['Profit']].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=4)

x_train.shape

x_test.shape

reg = LinearRegression()
reg.fit(x_train,y_train)

x_t=[[165349.20]]
y_pred = reg.predict(x_t)
y_pred

plt.scatter(x_train, y_train, color = 'black')
plt.plot(x_train, reg.predict(x_train), color = 'green')
plt.title('R&D Spend vs Profit (Training set)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()

plt.scatter(x_test, y_test, color = 'black')
plt.plot(x_train, reg.predict(x_train), color = 'green')
plt.title('R&D Spend vs Profit (Testing set)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()

# Multiple Linear Regression
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

multiple_regressor = LinearRegression()
multiple_regressor.fit(X_train, y_train)

y_pred = multiple_regressor.predict(X_test)

# Polynomial Linear Regression

df1 = pd.read_csv('/content/sample_data/Position_Salaries.csv')

df1.head()

X = df1.iloc[:, 1:2].values
y = df1.iloc[:, 2].values

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X, poly_regressor.predict(poly.fit_transform(X)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Logistic Regreassion
df = pd.read_csv('/content/sample_data/Admission_Predict.csv')

df.columns = df.columns.str.strip()

threshold = 0.5
df['Admit'] = (df['Chance of Admit'] > threshold).astype(int)

X = df.drop(['Chance of Admit', 'Admit'], axis=1)
y = df['Admit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

coefficients = model.coef_[0]
feature_names = X.columns
coeff_df = pd.DataFrame(coefficients, index=feature_names, columns=['Coefficient'])
print("Model Coefficients:\n", coeff_df)
