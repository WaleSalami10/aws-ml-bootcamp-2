import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate sample house data
np.random.seed(42)
house_sizes = np.random.uniform(1000, 4000, 100).reshape(-1, 1)
house_prices = 50 * house_sizes.flatten() + np.random.normal(0, 20000, 100) + 100000

# Train regression model
model = LinearRegression()
model.fit(house_sizes, house_prices)

# Generate predictions for smooth line
x_line = np.linspace(1000, 4000, 100).reshape(-1, 1)
y_pred = model.predict(x_line)

# Create chart
plt.figure(figsize=(10, 6))
plt.scatter(house_sizes, house_prices, alpha=0.6, color='blue', label='Actual Prices')
plt.plot(x_line, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('House Price Prediction - Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('house_prediction_regression.png', dpi=300, bbox_inches='tight')
plt.close()