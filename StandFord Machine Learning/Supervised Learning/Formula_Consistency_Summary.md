# Formula Consistency Summary

This document summarizes the key formula updates made to ensure consistency between the documentation and your actual algorithm implementations.

## Linear Regression Formulas (Consistent with Your Implementation)

### Cost Function
Your implementation uses:
```python
def calculate_cost(y, y_hat):
    return np.mean((y - y_hat) ** 2) / 2
```

**Mathematical Formula:**
$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

### Gradients
Your implementation uses:
```python
def compute_gradients(X, y, y_hat):
    m = len(y)
    dw = (1/m) * np.dot(X.T, (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)
    return dw, db
```

**Mathematical Formulas:**
$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

### Regularization Formulas

**Ridge (L2) Regularization:**
$$J_{ridge}(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 + \lambda \sum_{j=1}^{n} w_j^2$$

**Lasso (L1) Regularization:**
$$J_{lasso}(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 + \lambda \sum_{j=1}^{n} |w_j|$$

**Elastic Net:**
$$J_{elastic}(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 + \lambda_1 \sum_{j=1}^{n} |w_j| + \lambda_2 \sum_{j=1}^{n} w_j^2$$

## Logistic Regression Formulas (Consistent with Your Implementation)

### Cost Function
Your implementation uses:
```python
# Compute loss with numerical stability
epsilon = 1e-15  # Prevent log(0)
y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
loss = -(1/m) * np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
```

**Mathematical Formula:**
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$$

Where $\hat{y}^{(i)} = \sigma(w^T x^{(i)} + b)$ and $\sigma(z) = \frac{1}{1 + e^{-z}}$

### Gradients
Your implementation uses the same gradient computation as linear regression:
```python
def compute_gradients(X, y, y_pred):
    m = len(y)
    dw = (1/m) * np.dot(X.T, (y_pred - y))
    db = (1/m) * np.sum(y_pred - y)
    return dw, db
```

**Mathematical Formulas:**
$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

### Regularization Formulas

**Ridge Logistic Regression:**
$$J_{ridge}(w, b) = J(w, b) + \lambda \sum_{j=1}^{n} w_j^2$$

**Lasso Logistic Regression:**
$$J_{lasso}(w, b) = J(w, b) + \lambda \sum_{j=1}^{n} |w_j|$$

**Elastic Net Logistic Regression:**
$$J_{elastic}(w, b) = J(w, b) + \lambda_1 \sum_{j=1}^{n} |w_j| + \lambda_2 \sum_{j=1}^{n} w_j^2$$

Where $J(w, b)$ is the standard logistic regression cost function.

## Key Consistency Points

1. **Notation**: Using $w$ for weights and $b$ for bias (consistent with your code)
2. **Cost Function**: Using the exact same formulation as your `calculate_cost` functions
3. **Gradients**: Matching the exact gradient computation in your `compute_gradients` functions
4. **Regularization**: Using $\lambda$ for regularization strength (matching your `alpha` parameter)
5. **Numerical Stability**: Including epsilon clipping for logistic regression (as in your implementation)

## Implementation Notes

- Your linear regression uses MSE with the factor of 1/2 for mathematical convenience in derivatives
- Your logistic regression includes numerical stability with epsilon clipping
- Both implementations use the same gradient descent update rule
- Regularization terms are added to the base cost functions without modifying the gradient computation structure
- The bias term is not regularized in any of your implementations (which is standard practice)

This ensures that the mathematical formulas in the documentation exactly match what students will see when they examine your code implementations.