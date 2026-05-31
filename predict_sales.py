import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("Advertising.csv")


print(data.head())


print(data.info())

print(data.isnull().sum())

if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# Input features
X = data[['TV', 'Radio', 'Newspaper']]

y = data['Sales']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Predicted Sales:")
print(y_pred[:5])

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()