# Linear Regression Implementation from Scratch

This document explains the implementation of a multivariate linear regression model from scratch using NumPy. The model predicts house prices based on multiple features.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Implementation Steps](#implementation-steps)
4. [Mathematical Concepts](#mathematical-concepts)
5. [Code Structure](#code-structure)
6. [Model Evaluation](#model-evaluation)

## Overview

The implementation follows these key steps:
1. Data preparation and feature scaling
2. Model parameter initialization
3. Gradient descent optimization
4. Model evaluation and visualization

## Dataset

The model uses a housing dataset with the following features:
- `size_sqft`: Size of the house in square feet
- `bedrooms`: Number of bedrooms
- `age_years`: Age of the house in years
- Target variable: `price`

## Implementation Steps

### 1. Data Preparation
```python
# Load and prepare data
training_dataset = pd.read_csv('house_prices.csv')
X = training_dataset[['size_sqft', 'bedrooms', 'age_years']].values
y = training_dataset['price'].values
```

### 2. Feature Scaling
The implementation uses min-max normalization to scale features between 0 and 1:
```python
x1 = (x1 - x1.min()) / (x1.max() - x1.min())
x2 = (x2 - x2.min()) / (x2.max() - x2.min())
x3 = (x3 - x3.min()) / (x3.max() - x3.min())
```

### 3. Model Implementation

#### Cost Function
The Mean Squared Error (MSE) cost function:
```python
def calculate_cost(y, y_hat):
    return np.mean((y - y_hat) ** 2) / 2
```

#### Gradient Computation
```python
def compute_gradients(X, y, y_hat):
    m = len(y)
    dw = (1/m) * np.dot(X.T, (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)
    return dw, db
```

#### Parameter Updates
```python
def update_parameters(weights, bias, dw, db, learning_rate):
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db
    return weights, bias
```

## Mathematical Concepts

### Linear Regression Model
The model follows the equation:
$y = Xw + b$

where:
- $y$ is the predicted price
- $X$ is the feature matrix
- $w$ are the weights
- $b$ is the bias term

### Cost Function
Mean Squared Error (MSE):
$$J(w,b) = \frac{1}{2m} \sum_{i=1}^m (y^{(i)} - \hat{y}^{(i)})^2$$

### Gradient Descent
Updates parameters using:
$$w = w - \alpha \frac{\partial J}{\partial w}$$
$$b = b - \alpha \frac{\partial J}{\partial b}$$

where $\alpha$ is the learning rate.

## Code Structure

### Training Loop
The `learning_curve` function implements the main training loop:
```python
def learning_curve(X_z, y, learning_rate=0.01, epochs=1000):
    weights = np.random.randn(X_z.shape[1]) * 0.01
    bias = 0.0
    costs = []
    for _ in range(epochs):
        y_hat = np.dot(X_z, weights) + bias
        cost = calculate_cost(y, y_hat)
        dw, db = compute_gradients(X_z, y, y_hat)
        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)
    return weights, bias, costs
```

## Model Evaluation

The implementation includes two key metrics:

### R² Score (Coefficient of Determination)
```python
def r2_score(y_true, y_pred):
    residual_ss = np.sum((y_true - y_pred) ** 2)
    total_ss = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (residual_ss / total_ss)
```

### Mean Squared Error
```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

Interpretation of R² Score:
- 1.0: Perfect fit
- 0.7-0.9: Good fit
- Below 0.5: Model might need improvement

## Visualization

The implementation includes several visualizations:
1. Initial data distribution for each feature
2. Learning curve showing cost vs. iterations
3. Final predictions vs. actual values for each feature

These plots help in understanding:
- Data distribution
- Model convergence
- Prediction accuracy
- Feature relationships with the target variable