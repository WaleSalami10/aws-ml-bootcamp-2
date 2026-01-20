import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('house_prices.csv')

# Features and target
X = data[['size_sqft', 'bedrooms', 'age_years']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: ${mean_squared_error(y_test, y_pred, squared=False):,.0f}")
print(f"\nCoefficients:")
print(f"Size: ${model.coef_[0]:.2f} per sqft")
print(f"Bedrooms: ${model.coef_[1]:,.0f} per bedroom")
print(f"Age: ${model.coef_[2]:,.0f} per year")
print(f"Intercept: ${model.intercept_:,.0f}")