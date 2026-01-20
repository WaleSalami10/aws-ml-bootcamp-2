import numpy as np
import pandas as pd

# Generate sample data
np.random.seed(42)
n_samples = 1000

# Features
size = np.random.normal(2000, 500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
age = np.random.randint(0, 50, n_samples)

# Target prices with linear relationship + noise
prices = (100 * size + 50000 * bedrooms - 2000 * age + 100000 + 
          np.random.normal(0, 25000, n_samples))

# Create dataset
data = pd.DataFrame({
    'size_sqft': size,
    'bedrooms': bedrooms,
    'age_years': age,
    'price': prices
})

# Save to CSV
data.to_csv('house_prices.csv', index=False)
print(f"Generated {n_samples} house price samples")
print(data.head())